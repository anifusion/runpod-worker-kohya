"""
Handler for the generation of a fine tuned lora model.
"""

import os
import re
import select
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA

# Kohya tqdm reports moving-average loss as "avr_loss=..."; NaN runs must not ship weights.
TRAINING_LOSS_NAN_PATTERN = re.compile(r"avr_loss=nan\b", re.IGNORECASE)


def _run_training_subprocess(
    cmd_args: list,
    timeout_sec: int,
) -> tuple[int, bool, str]:
    """
    Run training with live stdout (RunPod logs), enforce timeout, detect NaN loss in output.
    Returns (returncode, saw_nan_in_output, output_tail).
    """
    if sys.platform == "win32":
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
        out = result.stdout or ""
        print(out, end="")
        saw_nan = bool(TRAINING_LOSS_NAN_PATTERN.search(out))
        return result.returncode, saw_nan, out[-2000:]

    proc = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    out_fd = proc.stdout.fileno()
    saw_nan = False
    text_buf = ""
    deadline = time.monotonic() + timeout_sec

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            raise subprocess.TimeoutExpired(cmd_args, timeout_sec)

        r, _, _ = select.select([proc.stdout], [], [], min(max(remaining, 0), 1.0))
        if proc.stdout in r:
            chunk = os.read(out_fd, 65536)
            if chunk:
                decoded = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(decoded)
                sys.stdout.flush()
                text_buf += decoded
                if len(text_buf) > 524288:
                    text_buf = text_buf[-262144:]
                if TRAINING_LOSS_NAN_PATTERN.search(text_buf):
                    saw_nan = True
        if proc.poll() is not None:
            break

    while True:
        chunk = os.read(out_fd, 65536)
        if not chunk:
            break
        decoded = chunk.decode("utf-8", errors="replace")
        sys.stdout.write(decoded)
        sys.stdout.flush()
        text_buf += decoded
        if len(text_buf) > 524288:
            text_buf = text_buf[-262144:]
        if TRAINING_LOSS_NAN_PATTERN.search(text_buf):
            saw_nan = True

    return proc.wait(), saw_nan, text_buf[-2000:]


def cuda_supports_bf16() -> bool:
    """Ampere+ (e.g. RTX 30/40/50) — mixed bf16 avoids full-fp16 overflow that often yields NaN loss."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        fn = getattr(torch.cuda, "is_bf16_supported", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def validate_url(url):
    """Validate that URL is safe and uses allowed schemes."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
        # Basic hostname validation (allow all valid hostnames including private IPs for legitimate use)
        if not parsed.hostname:
            return False
        return True
    except Exception:
        return False


def sanitize_filename(filename):
    """Sanitize filename to prevent path traversal."""
    # Remove any path separators and keep only the basename
    filename = os.path.basename(filename)
    # Remove any non-alphanumeric characters except dots, hyphens, underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    # Ensure filename doesn't start with dots
    filename = re.sub(r"^\.+", "", filename)
    return filename


def sanitize_string_param(param_value):
    """Sanitize string parameters to prevent command injection while preserving functionality."""
    if not isinstance(param_value, str):
        param_value = str(param_value)
    # Remove dangerous shell metacharacters but preserve single & for legitimate text
    # Remove: semicolons, backticks, $, pipes (|), shell operators (&&, ||), redirects (<>)
    param_value = re.sub(
        r"[;`$|<>]", "_", param_value
    )  # Remove individual dangerous chars
    param_value = re.sub(r"&&|\|\|", "_", param_value)  # Remove shell operators
    return param_value


def validate_numeric_param(param_value, min_val=None, max_val=None):
    """Validate numeric parameters."""
    try:
        num_val = float(param_value)
        if min_val is not None and num_val < min_val:
            return False
        if max_val is not None and num_val > max_val:
            return False
        return True
    except Exception:
        return False


def handler(job):
    job_input = job["input"]

    if "errors" in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {"error": job_input["errors"]}
    job_input = job_input["validated_input"]

    # Validate URLs
    if not validate_url(job_input["model_url"]):
        return {"error": "Invalid model URL"}
    if not validate_url(job_input["zip_url"]):
        return {"error": "Invalid zip URL"}

    # Validate and sanitize inputs
    model_url = job_input["model_url"]
    model_basename = sanitize_filename(os.path.basename(model_url))
    if not model_basename:
        return {"error": "Invalid model URL: could not derive a valid filename"}

    # Validate numeric parameters with appropriate ranges
    numeric_params = {
        "text_encoder_lr": (0.0, None),  # Learning rates should be positive
        "unet_lr": (0.0, None),  # Learning rates should be positive
        "learning_rate": (0.0, None),  # Learning rates should be positive
        "network_dim": (1, None),  # Dimensions should be positive integers
        "network_alpha": (1, None),  # Alpha should be positive
        "lr_scheduler_num_cycles": (1, None),  # Cycles should be positive
        "lr_warmup_steps": (0, None),  # Steps can be 0 or positive
        "train_batch_size": (1, None),  # Batch size should be positive
        "max_train_steps": (1, None),  # Steps should be positive
        "max_data_loader_num_workers": (0, None),  # Workers can be 0 or positive
        "http_log_every": (1, None),  # Log frequency should be positive
        "steps": (1, None),  # Steps should be positive
    }

    for param, (min_val, max_val) in numeric_params.items():
        if param in job_input and not validate_numeric_param(
            job_input[param], min_val, max_val
        ):
            return {
                "error": f"Invalid numeric parameter: {param} (must be >= {min_val})"
            }

    # Sanitize string parameters
    sanitized_params = {}
    string_params = [
        "instance_name",
        "class_name",
        "lr_scheduler",
        "optimizer_type",
        "http_log_endpoint",
        "http_log_name",
        "http_log_token",
    ]

    for param in string_params:
        if param in job_input:
            sanitized_params[param] = sanitize_string_param(job_input[param])

    VOLUME_DIR = "/runpod-volume"

    # Check if model exists in volume directory
    volume_model_path = os.path.join(VOLUME_DIR, model_basename)
    if os.path.exists(volume_model_path):
        print(f"Model found in volume, using cached version: {volume_model_path}")
        downloaded_model = {"file_path": volume_model_path}
    else:
        # Download the model file
        print(f"Downloading model from {model_url}")
        try:
            downloaded_model = rp_download.file(job_input["model_url"])
        except Exception as e:
            return {"error": f"Failed to download model: {str(e)}"}

        # Make sure we check if the volume directory exists, in that case just use the download file path
        if os.path.exists(VOLUME_DIR):
            print(f"Moving model to volume for caching: {volume_model_path}")
            try:
                shutil.copy(downloaded_model["file_path"], volume_model_path)
                original_file_path = downloaded_model["file_path"]

                # Update the file path to the volume directory
                downloaded_model["file_path"] = volume_model_path

                # Delete old file
                os.remove(original_file_path)
            except Exception as e:
                return {"error": f"Failed to cache model: {str(e)}"}

    # Download the zip file
    print(f"Downloading zip file from {job_input['zip_url']}")
    try:
        downloaded_input = rp_download.file(job_input["zip_url"])
    except Exception as e:
        return {"error": f"Failed to download zip file: {str(e)}"}

    # Clean up any stale training directory from previous jobs on this worker
    if os.path.exists("./training"):
        shutil.rmtree("./training")

    os.mkdir("./training")
    os.mkdir("./training/img")
    os.mkdir("./training/model")
    os.mkdir("./training/logs")

    # Create training data directory with kohya naming convention (repeats_instancename classname)
    image_extensions = [".jpg", ".jpeg", ".png"]
    allowed_extensions = image_extensions + [".txt"]
    safe_dir_name = f"{job_input['steps']}_{sanitized_params.get('instance_name', job_input['instance_name'])} {sanitized_params.get('class_name', job_input['class_name'])}"
    safe_dir_name = sanitize_string_param(safe_dir_name)
    flat_directory = f"./training/img/{safe_dir_name}"
    os.mkdir(flat_directory)

    for root, dirs, files in os.walk(downloaded_input["extracted_path"]):
        # Skip __MACOSX folder
        if "__MACOSX" in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(file_path, flat_directory)

    image_count = sum(
        1
        for f in os.listdir(flat_directory)
        if os.path.splitext(f)[1].lower() in image_extensions
    )
    if image_count == 0:
        return {
            "error": f"No training images found in extracted zip. Files in training dir: {os.listdir(flat_directory)}"
        }

    out_id = sanitize_string_param(job_input["out_id"] or job["id"])

    use_bf16 = cuda_supports_bf16()
    mixed_precision = "bf16" if use_bf16 else "fp16"
    # Keep saved LoRA in fp16 for smaller files; training uses mixed bf16/fp16 weights only.
    save_precision = "fp16"
    if use_bf16:
        print(
            "runpod-worker-kohya: mixed-precision bf16, no full_fp16 (stability best practice)"
        )
    else:
        print(
            "runpod-worker-kohya: mixed-precision fp16, no full_fp16 (bf16 not available on this GPU)"
        )

    # Build secure command arguments array (no shell injection possible)
    cmd_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        "sdxl_train_network.py",
        "--enable_bucket",
        "--pretrained_model_name_or_path",
        downloaded_model["file_path"],
        "--train_data_dir",
        "./training/img",
        "--resolution",
        "1024,1024",
        "--text_encoder_lr",
        str(job_input["text_encoder_lr"]),
        "--no_half_vae",
        "--mixed_precision",
        mixed_precision,
        "--save_precision",
        save_precision,
        "--gradient_checkpointing",
        "--unet_lr",
        str(job_input["unet_lr"]),
        "--network_dim",
        str(job_input["network_dim"]),
        "--network_alpha",
        str(job_input["network_alpha"]),
        "--lr_scheduler",
        sanitized_params.get("lr_scheduler", job_input["lr_scheduler"]),
        "--learning_rate",
        str(job_input["learning_rate"]),
        "--lr_scheduler_num_cycles",
        str(job_input["lr_scheduler_num_cycles"]),
        "--lr_warmup_steps",
        str(job_input["lr_warmup_steps"]),
        "--train_batch_size",
        str(job_input["train_batch_size"]),
        "--max_train_steps",
        str(job_input["max_train_steps"]),
        "--output_dir",
        "./training/model",
        "--output_name",
        out_id,
        "--max_data_loader_n_workers",
        str(job_input["max_data_loader_num_workers"]),
        "--caption_extension",
        "txt",
        "--save_model_as",
        "safetensors",
        "--network_module",
        "networks.lora",
        "--optimizer_type",
        sanitized_params.get("optimizer_type", job_input["optimizer_type"]),
        "--logging_dir",
        "./log",
        "--http-log",
        "--http-log-endpoint",
        sanitized_params.get("http_log_endpoint", job_input["http_log_endpoint"]),
        "--http-log-name",
        sanitized_params.get("http_log_name", job_input["http_log_name"]),
        "--http-log-token",
        sanitized_params.get("http_log_token", job_input["http_log_token"]),
        "--http-log-every",
        str(job_input["http_log_every"]),
        "--cache_latents",
        "--bucket_reso_steps",
        "64",
        "--bucket_no_upscale",
        "--tokenizer_cache_dir",
        "/tokenizer_cache",
    ]

    if job_input.get("v_parameterization"):
        cmd_args.append("--v_parameterization")
        print("runpod-worker-kohya: v-parameterization training enabled (v-pred base model)")
    if job_input.get("zero_terminal_snr"):
        cmd_args.append("--zero_terminal_snr")
        if job_input.get("v_parameterization"):
            print("runpod-worker-kohya: zero-terminal-SNR scheduler fix enabled")

    try:
        returncode, output_had_nan, output_tail = _run_training_subprocess(cmd_args, 3600)
    except subprocess.TimeoutExpired:
        return {"error": "Training process timed out"}
    except Exception as e:
        return {"error": f"Training process error: {str(e)}"}

    if returncode != 0:
        return {
            "error": f"Training process failed: {returncode}",
            "details": output_tail,
        }

    output_path = f"./training/model/{out_id}.safetensors"

    if output_had_nan:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return {
            "error": (
                "Training reported non-finite loss (NaN); checkpoint was discarded. "
                "Try lower learning rates, different images or captions, or a different base model."
            )
        }

    if not os.path.exists(output_path):
        return {"error": f"Training completed but output file not found: {output_path}"}

    job_s3_config = job.get("s3Config")

    try:
        uploaded_lora_url = upload_file_to_bucket(
            file_name=f"{out_id}.safetensors",
            file_location=output_path,
            bucket_creds=job_s3_config,
            # bucket_name=None if job_s3_config is None else job_s3_config['bucketName'],
            bucket_name="lora",
        )
    except Exception as e:
        return {"error": f"Failed to upload model: {str(e)}"}

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})
