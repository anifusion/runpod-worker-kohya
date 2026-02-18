"""
Handler for the generation of a fine tuned lora model.
"""

import os
import shutil
import subprocess
import re
from urllib.parse import urlparse

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA


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
        "fp16",
        "--save_precision",
        "fp16",
        "--full_fp16",
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
    ]

    try:
        subprocess.run(cmd_args, check=True, timeout=3600)  # 1 hour timeout
    except subprocess.TimeoutExpired:
        return {"error": "Training process timed out"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Training process failed: {e.returncode}"}
    except Exception as e:
        return {"error": f"Training process error: {str(e)}"}

    output_path = f"./training/model/{out_id}.safetensors"
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
