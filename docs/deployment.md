# Deployment and checkpoint cache

## Deployment

Runpod can build from a connected GitHub repository automatically after a push.

For a local build, use `docker build --platform linux/amd64 -t runpod-worker-kohya:local .`. The pinned Triton version does not provide a compatible arm64 wheel.

**Manual action after each push:** Open the endpoint's build history in Runpod. Confirm that the build for the pushed commit is marked `Completed` and the endpoint uses that image before running jobs.

## Model output storage

The worker uploads each trained LoRA to the exact bucket and key the app reads, then confirms the stored object's size before reporting success. A missing or wrong-size object fails the job. If the credentials cannot read the object, the worker logs that verification was unavailable and keeps the result.

The storage endpoint (`endpointUrl` in the job's bucket settings, or `BUCKET_ENDPOINT_URL`) is the account URL, optionally followed by one path segment naming the public bucket. The worker always connects to the account URL with path-style addressing and names the bucket explicitly:

| Output | Bucket | Key |
| --- | --- | --- |
| Public, endpoint names a bucket | That bucket | `lora/<id>.safetensors` |
| Public, endpoint names no bucket | `lora` | `<id>.safetensors` |
| Private (`private_output`) | `R2_PRIVATE_BUCKET` | `private-v1/lora/<id>.safetensors` |

Earlier versions passed the full endpoint URL, bucket path included, to the S3 client. Public uploads still landed in the right place, but every private upload was written into the endpoint's public bucket under `<private-bucket>/private-v1/lora/`, where the app never reads it. Public destinations are unchanged.

Without storage credentials the job fails, so the caller sees a failure instead of a model path nothing can read. `ALLOW_LOCAL_LORA_UPLOAD=1` restores the local copy for local development only.

## Checkpoint format and cache

Base checkpoints must use safetensors. Pickle/ZIP checkpoints and model URLs with `.ckpt`, `.pt`, `.pth`, or `.bin` suffixes are rejected before dataset preparation or training. Header inspection never invokes a pickle loader. Full tensor validation remains the safetensors reader's responsibility. A recognized header is not a trusted publisher signature or a full-file integrity check.

Model downloads stream raw bytes to a generated local filename. They do not use the SDK archive extractor or a server-provided filename. Failed streams are removed. HTTP failures stop model preparation. This download path does not use ambient netrc credentials or proxy settings. Redirect destination restrictions and download byte/deadline limits remain separate hardening requirements. Dataset ZIP handling is unchanged.

Every path passed to the training loader ends in the exact lowercase suffix `.safetensors`. Valid extensionless, uppercase-suffix and `.sft` downloads are renamed locally before cache publication. This also applies when no network volume exists or a volume write fails, so the fallback cannot select a pickle loader merely because the download lacks a suffix.

Each job deletes its local files when it ends, whether it succeeds or fails. This covers the model download (including one rejected by the format check, or used directly because the network volume was missing, full or failed to publish the cache), the downloaded training archive and its extracted folder, and the `./training` workspace with the copied images and outputs. Checkpoints are several gigabytes and warm workers run many jobs, so a retained copy would eventually fill the container disk. The archive, its extracted folder and the workspace hold the user's private images, so they must not outlive the job. Cleanup only removes entries directly inside the job's local download folder and the workspace, never the shared volume cache. A cleanup error is logged with stage `job_files_cleanup` or `training_cleanup` and does not change the job result.

Cache filenames use `checkpoint-v2-<SHA-256 of the full source URL>.safetensors`. The original hostname, path and query are not written into the filename or logged. Different hosts, paths and query variants do not share a basename cache. Stable source URLs retain stable cache keys. Changing signed URL parameters creates a different key. URL identity does not authenticate the downloaded bytes or detect a changed model at the same URL.

Legacy basename and extensionless caches are ignored and left intact. They are not copied into the new namespace because their provenance cannot be established. Cache reads reject symlinks and validate size, header and loader suffix. Publication stages and validates a copy before atomic replacement. A failed write preserves the previous cache and uses the validated local download. These controls do not defend against a party with arbitrary concurrent write access to the volume. Keep volume writers restricted.

## Safetensors cache transition

This one-time transition applies to an endpoint adopting the safetensors-only handler and `checkpoint-v2-` cache namespace. It ends when all claiming workers use the new image and supported-model acceptance passes. Later deployments retaining this contract need no separate cache migration.

Before rollout, verify each supported model resolves to safetensors through the endpoint's authorized access path. Header probes are compatibility evidence only. Check space for fresh downloads and source-bound caches alongside retained legacy files. Account for first-use download latency, provider rate limits and changed signed URL keys. Do not delete existing caches automatically.

After an authorized release, confirm the connected-repository build completes and all workers adopt the intended image. Verify extensionless safetensors, normal cached reuse, cache write failure, and an approved minimal GPU training job. Validate pickle/ZIP refusal with synthetic local tests, not executable payloads in production. Older workers retain unsafe loading and legacy cache behavior, so a mixed fleet is not complete containment. Rolling back to them reopens that behavior. Prefer a forward repair. Any cache cleanup requires a separate retention decision after old workers stop.

No environment-variable or sd-scripts revision change is required. For build or operational commands needing existing configuration, load env vars through the approved process (see `.env.example`).

Run synthetic worker regression checks with Python 3.11 or later: `python3 -B -m unittest discover -s tests`. These tests do not download models or run GPU training. No lint or formatter command is declared by this repository. Also run `bash -n builder/setup.sh` and `git diff --check`.

## HTTP dependency upgrade

The candidate pins aiohttp 3.14.5, aiohappyeyeballs 2.7.1, aiosignal 1.4.0, yarl 1.25.1, urllib3 2.8.0, requests 2.34.2, Jinja2 3.1.6, Werkzeug 3.1.9, httpcore 1.0.9 and certifi 2026.10.23. The accompanying aiohttp pins satisfy its dependency constraints. httpcore 1.0.9 requires h11 0.16 or later, removing the old h11 constraint. Both sd-scripts revisions are unchanged. The GPU stack is covered by the coordinated runtime upgrade. The setup script stops on failed commands and refreshes apt metadata before installation. This is partial security maintenance, not a clean vulnerability assessment.

November 12, 2026 verification: uv 0.13.8 resolved the complete registry requirement set for Linux amd64 and Python 3.10. The editable fork was omitted from this metadata-only check because its setup declares no dependencies. The initial HTTP-only check did not build an image. The completed Linux build is recorded below. No GPU run or deployment was performed. The constrained HTTP subset also passed Python 3.11 synthetic loopback checks for requests model streaming, HTTP error cleanup, aiohttp/httpx transport, Jinja2 escaping and yarl URL handling. That HTTP-only check did not verify Linux native wheels. The completed build and offline checks below cover Linux CPU behavior. GPU acceptance remains open.

For first adoption of these pins, use the deployment and safetensors transition checks in this document. Confirm installed package versions in the built image, scan its resolved Python and native packages, and verify downloads, startup, finite-loss training, callbacks and output before completing adoption. Subsequent deployments retaining these versions need no separate configuration step. Retain a verified compatible image for forward recovery. Do not use a pickle-accepting image as routine rollback.

The coordinated runtime upgrade is prepared locally as described next. Image/GPU acceptance and remaining dependency advisories are still open.

## Coordinated Python and GPU runtime upgrade

This candidate uses Python 3.11 in `/opt/venv`, PyTorch 2.15.0, torchvision 0.29.2, Pillow 13.0.0, Triton 3.9.0, bitsandbytes 0.50.2, filelock 4.2.10 and SymPy 1.14.0. NumPy 1.25.2 and the training-framework pins remain unchanged to limit compatibility changes. Both sd-scripts pins still select `59e616d1d1b5c52f2984d7004a96c2cff4b6ace3`. No training algorithm, job schema or callback change is included.

The NVIDIA base is CUDA 13.2.1 on Ubuntu 22.04, pinned by manifest digest. PyTorch's wheels supply CUDA 13.2.2 libraries. xformers 0.0.35 uses the explicit SHA-256-pinned CUDA 13.0 wheel from the official PyTorch index, not PyPI's CUDA 12.8 wheel. Its stable PyTorch ABI supports 2.10 and later. Native CUDA compatibility still requires execution on the intended GPU/driver. bitsandbytes includes its CUDA 13.2 backend. Do not mix the default CUDA 12 xformers binary into this image.

The headless training path does not require Tk. Published Python 3.11, development and venv packages were verified for Ubuntu 22.04. The venv PATH applies to pip, python3, accelerate, tokenizer caching and training subprocesses. The build checks the interpreter, runs `pip check`, and executes CPU compatibility checks against the actual pinned training code. It no longer force-reinstalls NumPy after resolving dependencies. The manifest remains authoritative.

### Local evidence and limits

November 12, 2026: registry resolution passed for Linux amd64/Python 3.11 with the selected GPU wheel. The NVIDIA manifest was verified to include amd64. CPU compatibility checks passed on both the previous torch 2.3.1/torchvision 0.18.1/Pillow 11.1.0 pair and the candidate under Python 3.11. They exercise the SDXL entry point, CLI contract, supported image formats and modes, resizing/cropping, gradient checkpointing, attention, tiny CLIP/VAE forward/backward passes, LoRA optimization and safetensors output. These are synthetic component checks, not a full SDXL training run or an output-quality guarantee. The 27 worker tests also pass. GPU checks are skipped on CPU and explicitly required by the release command. The complete Linux image build and offline CPU checks are verified below. Live GPU execution and deployment remain unverified.

### Completed Linux build

November 12, 2026: the full Linux amd64 image built successfully from source revision `8afa91a4d4086a0fb2020b9a6510a89881fc1fc5` as `runpod-worker-kohya:local`, local image ID `sha256:9d39edf406eeeac6eab147bae43be057d0c22682b079e175d122b5e797b9cbf3`, size 11,811,378,088 bytes. Python is 3.11.18. Dependency installation, `pip check`, the five CPU compatibility checks and both tokenizer-cache downloads passed. The scripts and editable library both resolve to the intended fork SHA. All 87 direct registry/wheel requirements match installed metadata, including the xformers archive hash. The built handler, schema, startup script and compatibility checks match the reviewed source.

With networking disabled in the final image, all 27 worker tests passed, the five component checks passed again, both tokenizer caches loaded and encoded a synthetic fixture, and the real handler imported, registered through a mocked SDK start function and rejected invalid synthetic input. Required-GPU mode failed specifically because CUDA hardware was absent, with no other compatibility failures. This verifies that CPU-only testing cannot falsely satisfy GPU acceptance. The emulated CPU emitted NNPACK hardware-fallback warnings, and diffusers emitted a deprecated pytree API warning. Neither caused a test failure. No checks or package pins were weakened.

Selected installed packages: torch 2.15.0, torchvision 0.29.2, Pillow 13.0.0, xformers 0.0.35, bitsandbytes 0.50.2, Triton 3.9.0, NumPy 1.25.2, CUDA toolkit metapackage 13.2.2, CUDA runtime package 13.2.86, cuDNN 9.26.0.51 and nvJitLink 13.5.37. The metapackage version does not mean every native package has the same minor version. Retain native-driver and GPU acceptance checks and scan the final image.

The first attempt stopped at apt upgrade because Docker lacked archive space. After authorized removal of 24 untagged images, the unchanged candidate built successfully. A remaining approved image was retained because Docker refused removal without force. No containers, volumes or tagged images were removed. That storage blocker is resolved for this build.

The subsequent compatible dependency patch and incremental scan are described next. GPU kernels, full SDXL training, output-quality/performance comparison, a clean release build of the final manifest and production adoption remain open. The image has not been pushed or deployed.

### Compatible support-library patch

The worker pins FastAPI 0.136.3, Starlette 1.3.1, fsspec 2026.6.0, protobuf 5.29.6, Rich 13.7.1 and setuptools 84.2.1. Unused pytorch-lightning, torchmetrics and lightning-utilities are removed. Training-framework versions and both fork pins remain unchanged. Framework advisory remediation requires a separately reviewed coordinated migration.

The build runs `python3 /checks/worker_compatibility.py` after copying the handler. This checks offline tokenizer IDs, the real SDK's synthetic request/validation path and handler registration/input refusal without provider connections. Runtime checks also cover SDXL checkpoint conversion, conditioning gradients and the training learning-rate scheduler.

November 12, 2026: the compatible manifest resolved independently for Linux amd64/Python 3.11. An incremental image built on the previously verified CUDA/Linux layers passed all 85 direct requirement checks, both fork pins, dependency consistency, 27 worker tests, seven CPU runtime checks and three installed-library checks. A merged-filesystem vulnerability scan completed. Remaining training-framework findings require coordinated follow-up. The incremental build avoids duplicate large layers but does not verify a clean build of the final Dockerfile. The endpoint's own build of the release commit is that clean build, and its build-time checks run there. Native GPU and full training acceptance happen through the post-release verification job in the one-time adoption steps.

### One-time adoption

Applies when moving from Python 3.10/CUDA 12.1/torch 2.3.1 to this candidate. The endpoint builds its image from the connected branch, so a push is the deployment. Adoption uses that normal build and verifies the running image with one real training job immediately after rollout, instead of a separate pre-release GPU environment.

1. Before pushing, raise the endpoint's minimum CUDA version from 12.1 to 13.2 and newer in its advanced settings. Images built for CUDA 12.1 also run on those hosts, so the current image keeps serving jobs. Without this change, workers placed on hosts with older drivers cannot start the new image and jobs wait in the queue. Afterwards, confirm the endpoint still finds workers for its GPU types. The candidate needs GPU compute capability 7.5 or later.
2. Push the adoption commits to the connected branch at a low-traffic time. Confirm the endpoint build completes. The build runs `pip check`, the CPU compatibility checks, tokenizer caching and the installed-library worker checks, so a failure there stops the release before any worker uses the new image.
3. As soon as the new build is active, run one small private training job from an owner or test account. In the job logs, confirm training ran on the GPU with finite loss, the job completed and the LoRA uploaded. Compare the LoRA's quality and the run time with an earlier training on the same base model. Check the other precision setting on a later job.
4. If that job fails or its output is clearly worse, roll back. Redeploy the previous release from the endpoint if the console offers it. Otherwise revert the adoption commits and push, so the endpoint rebuilds the previous runtime. Jobs that fail before the rollback completes return the worker's normal failure result.
5. After the verification job passes, confirm every worker runs the new image. Preserve the format/cache and private-output safeguards. Keep legacy caches until a separately approved cleanup. Record unresolved advisories separately.

For diagnosis, for example after a failed verification job, the image also contains a native GPU check that can run on any machine with a CUDA 13.2-capable NVIDIA driver:

```bash
docker run --rm --gpus all <image> \
  python3 /checks/runtime_compatibility.py --sd-scripts-root /sd-scripts --require-cuda
```

It checks the torch CUDA version and compute capability, and runs xformers attention/backpropagation plus a two-step AdamW8bit update. A run that reports the GPU test as skipped did not test the GPU.

No app payload migration or secret change is required. For provider commands needing existing credentials, load env vars through the approved process (see `.env.example`). This transition ends after the verification job passes and every worker runs the new image. Later deployments retaining the same runtime contract require no repeat CUDA-setting change.

Sources: [PyTorch package metadata](https://pypi.org/pypi/torch/2.15.0/json), [xformers stable ABI](https://github.com/facebookresearch/xformers/releases/tag/v0.0.34), [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation), [Pillow 13 changes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html).
