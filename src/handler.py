"""
Handler for the generation of a fine tuned lora model.
"""

import hashlib
import json
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlparse

import requests
import runpod
from boto3 import session as boto_session
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from runpod.serverless.utils.rp_upload import extract_region_from_url
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download
from tqdm_loggable.auto import tqdm

from rp_schema import INPUT_SCHEMA

LORA_UPLOAD_MAX_CONCURRENCY = 4
LORA_UPLOAD_MAX_POOL_CONNECTIONS = 16

# Kohya tqdm reports moving-average loss as "avr_loss=..."; NaN runs must not ship weights.
TRAINING_LOSS_NAN_PATTERN = re.compile(r"avr_loss=nan\b", re.IGNORECASE)

# Use /tmp so accelerate config is never blocked by a full Runpod volume (HF_HOME).
ACCELERATE_CONFIG_PATH = "/tmp/anifusion_accelerate_default_config.yaml"
# SDXL checkpoints are multi-GB; tiny files are failed/partial volume caches.
_MIN_VOLUME_CHECKPOINT_BYTES = 1_000_000
_TRAINING_DIAGNOSTIC_TAIL_CHARS = 131072


class TrainingSubprocessTimeout(Exception):
    def __init__(self, safe_tail):
        super().__init__("Training subprocess exceeded its time limit")
        self.safe_tail = safe_tail


def log_worker_failure(stage, error=None, *, tail=None, credential_boundary=False, exit_code=None,
                       job_id=None, code=None, parameter=None):
    """Build a bounded diagnostic for restricted worker and application logs."""
    try:
        diagnostic = {"stage": stage}
        if isinstance(job_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            diagnostic["job_id"] = job_id
        if isinstance(exit_code, int):
            diagnostic["exit_code"] = exit_code
        if code in {"InvalidModelUrl", "InvalidZipUrl", "InvalidNumericParameter",
                    "PrivateStorageUnavailable", "InvalidOutputId"}:
            diagnostic["code"] = code
        if isinstance(parameter, str) and re.fullmatch(r"[a-z_]{1,40}", parameter):
            diagnostic["parameter"] = parameter
        if isinstance(error, BaseException):
            diagnostic["type"] = type(error).__name__[:80]
            try:
                errno = getattr(error, "errno", None)
            except Exception:
                errno = None
            if isinstance(errno, int) and 0 <= errno <= 4095:
                diagnostic["errno"] = errno
            try:
                response = getattr(error, "response", None)
            except Exception:
                response = None
            if isinstance(response, dict):
                metadata = response.get("ResponseMetadata")
                if isinstance(metadata, dict):
                    status = metadata.get("HTTPStatusCode")
                    if isinstance(status, int) and 100 <= status <= 599:
                        diagnostic["status"] = status
                    request_id = metadata.get("RequestId")
                    if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,128}", request_id):
                        diagnostic["provider_request_id"] = request_id
                provider_error = response.get("Error")
                provider_code = provider_error.get("Code") if isinstance(provider_error, dict) else None
                if isinstance(provider_code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", provider_code):
                    diagnostic["code"] = provider_code
            try:
                status = getattr(error, "status_code", None)
            except Exception:
                status = None
            if isinstance(status, int) and 100 <= status <= 599:
                diagnostic["status"] = status
            if not credential_boundary:
                diagnostic["message"] = str(error)[:16000]
                diagnostic["frames"] = [
                    f"{frame.name}:{frame.lineno}"
                    for frame in traceback.extract_tb(error.__traceback__)[-6:]
                ]
                cause = error.__cause__ or error.__context__
                if isinstance(cause, BaseException) and cause is not error:
                    diagnostic["cause_type"] = type(cause).__name__[:80]
                    diagnostic["cause_message"] = str(cause)[:8000]
        if isinstance(tail, str) and not credential_boundary:
            diagnostic["worker_tail"] = tail[-_TRAINING_DIAGNOSTIC_TAIL_CHARS:]
        worker_log_diagnostic = dict(diagnostic)
        for key in ("message", "cause_message", "worker_tail"):
            value = worker_log_diagnostic.get(key)
            if isinstance(value, str) and len(value) > 750:
                worker_log_diagnostic[key] = value[-750:]
                worker_log_diagnostic[key + "_chars"] = len(value)
        print("runpod-worker-kohya: failure " + json.dumps(worker_log_diagnostic, ensure_ascii=True))
        safe_stage = stage if isinstance(stage, str) and re.fullmatch(r"[a-z_]{1,40}", stage) else "unknown"
        safe_event = {"event": "worker_failure", "stage": safe_stage}
        if "job_id" in diagnostic:
            safe_event["job_id"] = diagnostic["job_id"]
        if "status" in diagnostic:
            safe_event["status"] = diagnostic["status"]
        print("runpod-worker-kohya: safe_event " + json.dumps(safe_event, ensure_ascii=True))
        return diagnostic
    except Exception:
        safe_stage = stage if isinstance(stage, str) and re.fullmatch(r"[a-z_]{1,40}", stage) else "unknown"
        safe_job_id = job_id if isinstance(job_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id) else None
        print(f"runpod-worker-kohya: failure stage={safe_stage}" + (f" job_id={safe_job_id}" if safe_job_id else ""))
        safe_event = {"event": "worker_failure", "stage": safe_stage}
        if safe_job_id:
            safe_event["job_id"] = safe_job_id
        print("runpod-worker-kohya: safe_event " + json.dumps(safe_event, ensure_ascii=True))
        return {"stage": safe_stage, **({"job_id": safe_job_id} if safe_job_id else {})}


def safe_training_failure_label(tail: str, exit_code: int) -> str:
    """Preserve existing webhook guidance without returning the worker log tail."""
    lower = tail.lower()
    if "illegal memory access" in lower:
        return f"Training process failed: {exit_code}"
    if "no training images found" in lower or "training zip would contain no images" in lower:
        return "No training images found"
    if "out of memory" in lower:
        return "CUDA out of memory"
    if "timeout" in lower or "timed out" in lower:
        return "Training timed out"
    if "disk quota exceeded" in lower or "errno 122" in lower:
        return "Disk quota exceeded"
    if ("nonetype" in lower and "get" in lower) or "load_config_from_file" in lower or "failed to write accelerate config" in lower:
        return "Failed to write accelerate config"
    if any(marker in lower for marker in ("unpicklingerror", "invalid load key", "model_url", "model download", "failed to download model")):
        return "Base model download failed"
    if any(marker in lower for marker in ("corrupt", "invalid", "truncated")):
        return "Training data invalid"
    if any(marker in lower for marker in ("file name too long", "errno 36", "nametoolong")):
        return "File name too long"
    return f"Training process failed: {exit_code}"


def ensure_accelerate_config(mixed_precision: str) -> str:
    """Write a valid accelerate launch config and return its path."""
    if mixed_precision not in ("bf16", "fp16"):
        mixed_precision = "fp16"
    config_body = f"""compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: {mixed_precision}
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    try:
        with open(ACCELERATE_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            config_file.write(config_body)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write accelerate config at {ACCELERATE_CONFIG_PATH}: {exc}"
        ) from exc
    return ACCELERATE_CONFIG_PATH


def _run_training_subprocess(
    cmd_args: list,
    timeout_sec: int,
) -> tuple[int, bool, str]:
    """
    Run training, enforce timeout, detect NaN loss, and keep raw stdout in memory.
    Returns (returncode, saw_nan_in_output, output_tail).
    """
    # Provider credentials can appear in subprocess output. Keep raw output in
    # process memory and expose only a bounded tail through log_worker_failure.
    sensitive_args = []
    for option in ("--http-log-token", "--http-log-endpoint"):
        try:
            value = cmd_args[cmd_args.index(option) + 1]
        except (ValueError, IndexError):
            continue
        if isinstance(value, str) and value and value != "none":
            sensitive_args.append(value)

    def redact_known_credentials(output: str) -> str:
        for value in sensitive_args:
            output = output.replace(value, "[credential-omitted]")
        return output

    if sys.platform == "win32":
        try:
            result = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or error.output or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            raise TrainingSubprocessTimeout(redact_known_credentials(output)[-_TRAINING_DIAGNOSTIC_TAIL_CHARS:]) from None
        out = result.stdout or ""
        saw_nan = bool(TRAINING_LOSS_NAN_PATTERN.search(out))
        return result.returncode, saw_nan, redact_known_credentials(out)[-_TRAINING_DIAGNOSTIC_TAIL_CHARS:]

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
            raise TrainingSubprocessTimeout(redact_known_credentials(text_buf)[-_TRAINING_DIAGNOSTIC_TAIL_CHARS:])

        r, _, _ = select.select([proc.stdout], [], [], min(max(remaining, 0), 1.0))
        if proc.stdout in r:
            chunk = os.read(out_fd, 65536)
            if chunk:
                decoded = chunk.decode("utf-8", errors="replace")
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
        text_buf += decoded
        if len(text_buf) > 524288:
            text_buf = text_buf[-262144:]
        if TRAINING_LOSS_NAN_PATTERN.search(text_buf):
            saw_nan = True

    return proc.wait(), saw_nan, redact_known_credentials(text_buf)[-_TRAINING_DIAGNOSTIC_TAIL_CHARS:]


def cuda_supports_bf16() -> bool:
    """Ampere+ (e.g. RTX 30/40/50) · mixed bf16 avoids full-fp16 overflow that often yields NaN loss."""
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


_CHECKPOINT_SUFFIXES = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
)

_MAX_SAFE_TENSORS_HEADER = 100 * 1024 * 1024


def checkpoint_leaf_from_url(model_url: str) -> str:
    """Sanitized URL path basename (may already include .safetensors, .ckpt, etc.)."""
    path = urlparse(model_url).path
    base = sanitize_filename(os.path.basename(path))
    return base or "base_model"


def leaf_has_checkpoint_extension(leaf: str) -> bool:
    lower = leaf.lower()
    return any(lower.endswith(sfx) for sfx in _CHECKPOINT_SUFFIXES)


def sniff_checkpoint_extension(local_path: str) -> str:
    """Recognize safetensors without invoking a pickle or tensor loader."""
    with open(local_path, "rb") as f:
        head = f.read(8)
    if len(head) < 8:
        raise ValueError("file too small")
    if head[:2] == b"\x1f\x8b":
        raise ValueError("gzip prefix (need raw checkpoint bytes)")
    if head[:4] == b"PK\x03\x04":
        raise ValueError("pickle and zip checkpoints are not supported")
    # A safetensors header length of 128 starts with 0x80, so that byte alone is not a pickle signature.
    hlen = struct.unpack("<Q", head)[0]
    if 1 <= hlen <= _MAX_SAFE_TENSORS_HEADER:
        with open(local_path, "rb") as f:
            f.seek(8)
            raw = f.read(hlen)
        if len(raw) >= hlen:
            try:
                meta = json.loads(raw.decode("utf-8"))
                if isinstance(meta, dict):
                    return ".safetensors"
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                pass
    raise ValueError("checkpoint must use safetensors")


def checkpoint_matches_path(path: str) -> bool:
    """Reject incomplete caches and files whose contents disagree with the loader suffix."""
    try:
        if os.path.getsize(path) < _MIN_VOLUME_CHECKPOINT_BYTES:
            return False
        detected = sniff_checkpoint_extension(path)
    except (OSError, ValueError):
        return False
    suffix = os.path.splitext(path)[1]
    return suffix == ".safetensors" and detected == ".safetensors"


def checkpoint_cache_path(volume_dir: str, model_url: str) -> str:
    """Separate sources, including query variants, without persisting raw URLs."""
    identity = hashlib.sha256(model_url.encode("utf-8")).hexdigest()
    return os.path.join(volume_dir, "checkpoint-v2-" + identity + ".safetensors")


def resolve_volume_checkpoint_path(volume_dir: str, model_url: str) -> str | None:
    """Ignore legacy basename caches whose source identity cannot be established."""
    path = checkpoint_cache_path(volume_dir, model_url)
    if os.path.isfile(path) and not os.path.islink(path) and checkpoint_matches_path(path):
        return path
    return None


def download_checkpoint(model_url: str) -> dict:
    """Stream raw model bytes without SDK archive extraction or server-supplied filenames."""
    os.makedirs("job_files", exist_ok=True)
    fd, path = tempfile.mkstemp(dir="job_files", suffix=".download")
    try:
        with os.fdopen(fd, "wb") as output, requests.Session() as session:
            # Do not attach ambient netrc credentials to model hosts or redirects.
            session.trust_env = False
            with session.get(model_url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        return {"file_path": os.path.abspath(path)}
    except Exception:
        os.unlink(path)
        raise


def normalize_downloaded_checkpoint(downloaded_path: str) -> str:
    """Ensure fallback downloads select the safetensors loader without a volume."""
    if os.path.getsize(downloaded_path) < _MIN_VOLUME_CHECKPOINT_BYTES:
        raise ValueError("downloaded checkpoint is too small")
    sniff_checkpoint_extension(downloaded_path)
    if os.path.splitext(downloaded_path)[1] == ".safetensors":
        return downloaded_path
    fd, normalized = tempfile.mkstemp(suffix=".safetensors", dir=os.path.dirname(downloaded_path) or ".")
    os.close(fd)
    try:
        os.replace(downloaded_path, normalized)
    except OSError:
        os.unlink(normalized)
        raise
    return normalized


def finalize_volume_checkpoint_dest(
    volume_dir: str, model_url: str, downloaded_path: str
) -> str:
    """Validate downloaded bytes before choosing a source-bound cache path."""
    leaf = checkpoint_leaf_from_url(model_url)
    if os.path.getsize(downloaded_path) < _MIN_VOLUME_CHECKPOINT_BYTES:
        raise ValueError("downloaded checkpoint is too small")
    sniff_checkpoint_extension(downloaded_path)
    if leaf_has_checkpoint_extension(leaf) and os.path.splitext(leaf)[1].lower() not in (".safetensors", ".sft"):
        raise ValueError("checkpoint URL selects an unsupported format")
    return checkpoint_cache_path(volume_dir, model_url)


def cache_downloaded_checkpoint(source_path: str, dest_path: str, volume_dir: str) -> None:
    """Publish a validated copy atomically so concurrent jobs never read partial bytes."""
    with tempfile.TemporaryDirectory(dir=volume_dir) as staging_dir:
        staged = os.path.join(staging_dir, os.path.basename(dest_path))
        shutil.copy2(source_path, staged)
        if os.path.getsize(staged) != os.path.getsize(source_path) or not checkpoint_matches_path(staged):
            raise ValueError("staged checkpoint is incomplete or invalid")
        os.replace(staged, dest_path)


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


def storage_credentials(bucket_creds=None):
    """Return (endpoint_url, access_key_id, secret_access_key) from the job or environment."""
    if bucket_creds:
        return bucket_creds.get("endpointUrl"), bucket_creds.get("accessId"), bucket_creds.get("accessSecret")
    return (
        os.environ.get("BUCKET_ENDPOINT_URL"),
        os.environ.get("BUCKET_ACCESS_KEY_ID"),
        os.environ.get("BUCKET_SECRET_ACCESS_KEY"),
    )


def split_storage_endpoint(endpoint_url):
    """Split an S3 endpoint into its account URL and an optional bucket path.

    Deployments may configure the endpoint as `https://<account-host>/<bucket>`.
    Passing that URL to the S3 client turns every requested bucket into a folder
    inside `<bucket>`, so a private upload to another bucket silently lands in the
    configured one. Uploads therefore use the account URL and name buckets
    explicitly. Returns (account_url, bucket or None).
    """
    parsed = urlparse(endpoint_url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("model storage endpoint must be an http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("model storage endpoint must not contain a query or fragment")
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) > 1:
        raise ValueError("model storage endpoint path may name at most one bucket")
    return f"{parsed.scheme}://{parsed.netloc}", (segments[0] if segments else None)


def lora_output_destination(out_id, private_output, endpoint_bucket=None):
    """Return the (bucket, key) the app reads for this training's LoRA.

    Private weights go to R2_PRIVATE_BUCKET under versioned `private-v1/lora/` keys.
    Public weights go to `lora/<id>.safetensors` in the bucket named by the storage
    endpoint path, or to the legacy `lora` bucket when the endpoint names none.
    """
    if private_output:
        bucket = os.environ.get("R2_PRIVATE_BUCKET")
        if not bucket:
            raise RuntimeError("private_model_storage_unavailable")
        return bucket, f"private-v1/lora/{out_id}.safetensors"
    if endpoint_bucket:
        return endpoint_bucket, f"lora/{out_id}.safetensors"
    return "lora", f"{out_id}.safetensors"


def upload_lora_to_bucket(
    file_name: str,
    file_location: str,
    bucket_creds=None,
    bucket_name: str = "lora",
    require_remote: bool = False,
) -> str:
    """
    Upload a trained LoRA to object storage with bounded multipart concurrency so
    urllib3 does not warn about connection pool exhaustion on high-core hosts.

    `bucket_name` and `file_name` must be the exact bucket and key the app reads
    (see lora_output_destination). The upload is verified by size before the job
    reports success, so a misrouted or partial upload fails the job instead.
    """
    transfer_config = TransferConfig(
        multipart_threshold=1024 * 25,
        max_concurrency=LORA_UPLOAD_MAX_CONCURRENCY,
        multipart_chunksize=1024 * 25,
        use_threads=True,
    )

    endpoint_url, access_key_id, secret_access_key = storage_credentials(bucket_creds)

    if not (endpoint_url and access_key_id and secret_access_key):
        if require_remote:
            raise RuntimeError("private_model_credentials_unavailable")
        # A local copy is never readable by the app, so it is only for local
        # development, never for a deployed worker.
        if os.environ.get("ALLOW_LOCAL_LORA_UPLOAD") != "1":
            raise RuntimeError("model_storage_credentials_unavailable")
        print("No bucket endpoint set, saving to local upload folder")
        os.makedirs("local_upload", exist_ok=True)
        local_upload_location = f"local_upload/{file_name}"
        shutil.copyfile(file_location, local_upload_location)
        return local_upload_location

    account_url, _ = split_storage_endpoint(endpoint_url)
    region = extract_region_from_url(account_url)
    bucket_session = boto_session.Session()
    boto_client = bucket_session.client(
        "s3",
        endpoint_url=account_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=LORA_UPLOAD_MAX_POOL_CONNECTIONS,
            # Name buckets in the path explicitly so the destination never
            # depends on the client library's default addressing style.
            s3={"addressing_style": "path"},
        ),
        region_name=region,
    )

    file_size = os.path.getsize(file_location)
    with tqdm(total=file_size, unit="B", unit_scale=True, desc="model upload") as progress_bar:
        boto_client.upload_file(
            Filename=file_location,
            Bucket=bucket_name,
            Key=file_name,
            Config=transfer_config,
            ExtraArgs={"CacheControl": "private, no-store"} if require_remote else {},
            Callback=progress_bar.update,
        )

    try:
        stored = boto_client.head_object(Bucket=bucket_name, Key=file_name)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in ("403", "AccessDenied", "Forbidden"):
            raise RuntimeError("uploaded_model_verification_failed") from None
        # Write-only credentials cannot confirm the object. The upload call
        # already succeeded, so keep the result rather than fail every job.
        print("runpod-worker-kohya: upload verification unavailable (read access denied)")
    else:
        if stored.get("ContentLength") != file_size:
            raise RuntimeError("uploaded_model_verification_failed")

    return boto_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": file_name},
        ExpiresIn=604800,
    )


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


def remove_local_job_files(paths, job_id=None):
    """Delete a job's local downloads, never the shared volume cache.

    Base checkpoints are several gigabytes and warm workers run many jobs, so a
    download left behind after a failed check or an unusable volume fills the
    container disk and fails every later job on that worker. Training archives
    and their extracted folders hold the user's private images, so they must not
    outlive the job either. Only entries directly inside job_files are removed.
    """
    if not paths:
        return
    download_dir = os.path.realpath("job_files")
    for path in paths:
        try:
            real_path = os.path.realpath(path)
            if os.path.dirname(real_path) != download_dir:
                continue
            if os.path.isdir(real_path) and not os.path.islink(path):
                shutil.rmtree(real_path)
            else:
                os.remove(real_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            log_worker_failure("job_files_cleanup", error, credential_boundary=True, job_id=job_id)


def remove_training_workspace(job_id=None):
    """Delete the copied training images and outputs once the job has returned."""
    try:
        shutil.rmtree("./training")
    except FileNotFoundError:
        pass
    except OSError as error:
        log_worker_failure("training_cleanup", error, credential_boundary=True, job_id=job_id)


def handler(job):
    """Run one training job and always remove its local files afterwards."""
    local_job_files = []
    job_id = job.get("id") if isinstance(job, dict) else None
    try:
        return _handle_job(job, local_job_files)
    finally:
        remove_local_job_files(local_job_files, job_id=job_id)
        if local_job_files:
            # Only jobs that reached the download stage created a workspace.
            remove_training_workspace(job_id=job_id)


def _handle_job(job, local_job_files):
    try:
        job_input = job["input"]
        job_input = validate(job_input, INPUT_SCHEMA)
    except Exception as error:
        diagnostic = log_worker_failure("input_validation", error, credential_boundary=True, job_id=job.get("id"))
        return {"error": "Invalid training input", "failure_stage": "input_validation", "diagnostic": diagnostic}

    if not isinstance(job_input, dict) or "errors" in job_input or not isinstance(job_input.get("validated_input"), dict):
        diagnostic = log_worker_failure("input_validation", job_id=job.get("id"))
        return {"error": "Invalid training input", "failure_stage": "input_validation", "diagnostic": diagnostic}
    job_input = job_input["validated_input"]

    if job_input.get("private_output") is True:
        try:
            lora_output_destination("validation", True)
            credentials = job.get("s3Config")
            configured = all(credentials.get(key) for key in ("endpointUrl", "accessId", "accessSecret")) if credentials else all(
                os.environ.get(key) for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
            )
            if not configured:
                raise RuntimeError("private_model_credentials_unavailable")
        except Exception:
            diagnostic = log_worker_failure("input_validation", job_id=job.get("id"), code="PrivateStorageUnavailable")
            return {"error": "Private model storage is unavailable", "failure_stage": "input_validation", "diagnostic": diagnostic}
        # The id becomes an object key and a local file name, so reject separators and dot segments.
        private_out_id = job_input.get("out_id") or job.get("id")
        if not isinstance(private_out_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", private_out_id):
            diagnostic = log_worker_failure("input_validation", job_id=job.get("id"), code="InvalidOutputId")
            return {"error": "Invalid output id", "failure_stage": "input_validation", "diagnostic": diagnostic}

    # Validate URLs
    if not validate_url(job_input["model_url"]):
        diagnostic = log_worker_failure("input_validation", job_id=job.get("id"), code="InvalidModelUrl")
        return {"error": "Invalid model URL", "failure_stage": "input_validation", "diagnostic": diagnostic}
    if not validate_url(job_input["zip_url"]):
        diagnostic = log_worker_failure("input_validation", job_id=job.get("id"), code="InvalidZipUrl")
        return {"error": "Invalid zip URL", "failure_stage": "input_validation", "diagnostic": diagnostic}

    # Validate and sanitize inputs
    model_url = job_input["model_url"]

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
            diagnostic = log_worker_failure("input_validation", job_id=job.get("id"),
                                            code="InvalidNumericParameter", parameter=param)
            return {
                "error": f"Invalid numeric parameter: {param} (must be >= {min_val})",
                "failure_stage": "input_validation",
                "diagnostic": diagnostic,
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

    # Only source-bound safetensors caches may reach the training loader.
    volume_model_path = resolve_volume_checkpoint_path(VOLUME_DIR, model_url)
    if volume_model_path:
        print("Model found in volume, using cached version")
        downloaded_model = {"file_path": volume_model_path}
    else:
        # Download the model file
        print("runpod-worker-kohya: downloading model")
        try:
            downloaded_model = download_checkpoint(job_input["model_url"])
        except Exception as e:
            diagnostic = log_worker_failure("model_download", e, credential_boundary=True, job_id=job.get("id"))
            return {"error": "Failed to download model", "failure_stage": "model_download", "diagnostic": diagnostic}
        # handler deletes recorded local downloads when the job ends. The shared
        # volume copy is never recorded, so the cache is kept.
        local_job_files.append(downloaded_model["file_path"])

        try:
            volume_model_path = finalize_volume_checkpoint_dest(
                VOLUME_DIR, model_url, downloaded_model["file_path"]
            )
            downloaded_model["file_path"] = normalize_downloaded_checkpoint(downloaded_model["file_path"])
            local_job_files.append(downloaded_model["file_path"])
        except (OSError, ValueError) as e:
            diagnostic = log_worker_failure("model_checkpoint", e, credential_boundary=True, job_id=job.get("id"))
            return {"error": "Downloaded file is not a usable base checkpoint", "failure_stage": "model_checkpoint", "diagnostic": diagnostic}
        if os.path.isdir(VOLUME_DIR):
            print("Moving model to volume for caching")
            try:
                cache_downloaded_checkpoint(downloaded_model["file_path"], volume_model_path, VOLUME_DIR)
            except Exception as e:
                # A full or unavailable volume must not prevent training from the download.
                log_worker_failure("volume_cache", e, credential_boundary=True, job_id=job.get("id"))
                print("runpod-worker-kohya: training with downloaded model")
            else:
                original_file_path = downloaded_model["file_path"]
                downloaded_model["file_path"] = volume_model_path
                if original_file_path != volume_model_path:
                    try:
                        os.remove(original_file_path)
                    except OSError:
                        pass

    # Download the zip file
    print("runpod-worker-kohya: downloading training archive")
    try:
        downloaded_input = rp_download.file(job_input["zip_url"])
    except Exception as e:
        diagnostic = log_worker_failure("dataset_download", e, credential_boundary=True, job_id=job.get("id"))
        return {"error": "Failed to download training archive", "failure_stage": "dataset_download", "diagnostic": diagnostic}
    # The archive and its extracted folder hold the user's private images.
    local_job_files.append(downloaded_input["file_path"])
    if downloaded_input.get("extracted_path"):
        local_job_files.append(downloaded_input["extracted_path"])

    # Clean up any stale training directory from previous jobs on this worker
    if os.path.exists("./training"):
        shutil.rmtree("./training")

    os.mkdir("./training")
    os.mkdir("./training/img")
    os.mkdir("./training/model")
    os.mkdir("./training/logs")

    # Create training data directory with kohya naming convention (repeats_instancename classname).
    # Supported training zip image extensions. If an extension is omitted here, only captions (.txt)
    # copy through and training fails with "No training images found in extracted zip".
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"]
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
        diagnostic = log_worker_failure("dataset_prepare", job_id=job.get("id"))
        return {
            "error": "No training images found in archive",
            "failure_stage": "dataset_prepare",
            "diagnostic": diagnostic,
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

    try:
        accelerate_config_path = ensure_accelerate_config(mixed_precision)
    except RuntimeError as e:
        diagnostic = log_worker_failure("worker_configuration", e, job_id=job.get("id"))
        return {"error": "Failed to write accelerate config", "failure_stage": "worker_configuration", "diagnostic": diagnostic}

    # Build secure command arguments array (no shell injection possible)
    cmd_args = [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config_path,
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
    except TrainingSubprocessTimeout as e:
        diagnostic = log_worker_failure("training_subprocess", e, tail=e.safe_tail, job_id=job.get("id"))
        return {"error": "Training process timed out", "failure_stage": "training_subprocess", "diagnostic": diagnostic}
    except Exception as e:
        diagnostic = log_worker_failure("training_subprocess", e, credential_boundary=True, job_id=job.get("id"))
        return {"error": "Training process failed", "failure_stage": "training_subprocess", "diagnostic": diagnostic}

    if returncode != 0:
        diagnostic = log_worker_failure("training_subprocess", tail=output_tail, exit_code=returncode, job_id=job.get("id"))
        return {
            "error": safe_training_failure_label(output_tail[-2000:], returncode),
            "failure_stage": "training_subprocess",
            "exit_code": returncode,
            "diagnostic": diagnostic,
        }

    output_path = f"./training/model/{out_id}.safetensors"

    if output_had_nan:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        diagnostic = log_worker_failure("training_subprocess", tail=output_tail, exit_code=0, job_id=job.get("id"))
        return {
            "error": (
                "Training reported non-finite loss (NaN); checkpoint was discarded. "
                "Try lower learning rates, different images or captions, or a different base model."
            ),
            "failure_stage": "training_subprocess",
            "exit_code": 0,
            "diagnostic": diagnostic,
        }

    if not os.path.exists(output_path):
        diagnostic = log_worker_failure("training_output", job_id=job.get("id"))
        return {"error": "Training completed without an output model", "failure_stage": "training_output", "diagnostic": diagnostic}

    job_s3_config = job.get("s3Config")

    try:
        private_output = job_input.get("private_output") is True
        endpoint_url, _, _ = storage_credentials(job_s3_config)
        _, endpoint_bucket = split_storage_endpoint(endpoint_url) if endpoint_url else (None, None)
        output_bucket, output_key = lora_output_destination(out_id, private_output, endpoint_bucket)
        uploaded_lora_url = upload_lora_to_bucket(
            file_name=output_key,
            file_location=output_path,
            bucket_creds=job_s3_config,
            bucket_name=output_bucket,
            require_remote=private_output,
        )
    except Exception as e:
        diagnostic = log_worker_failure("model_upload", e, credential_boundary=True, job_id=job.get("id"))
        return {"error": "Failed to upload model", "failure_stage": "model_upload", "diagnostic": diagnostic}

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})
