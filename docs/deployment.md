# Deployment and checkpoint cache

## Deployment

Runpod can build from a connected GitHub repository automatically after a push.

For a local build, use `docker build --platform linux/amd64 -t runpod-worker-kohya:local .`. The pinned Triton version does not provide a compatible arm64 wheel.

**Manual action after each push:** Open the endpoint's build history in Runpod. Confirm that the build for the pushed commit is marked `Completed` and the endpoint uses that image before running jobs.

## Checkpoint format and cache

Base checkpoints must use safetensors. Pickle/ZIP checkpoints and model URLs with `.ckpt`, `.pt`, `.pth`, or `.bin` suffixes are rejected before dataset preparation or training. Header inspection never invokes a pickle loader. Full tensor validation remains the safetensors reader's responsibility. A recognized header is not a trusted publisher signature or a full-file integrity check.

Model downloads stream raw bytes to a generated local filename. They do not use the SDK archive extractor or a server-provided filename. Failed streams are removed. HTTP failures stop model preparation. This download path does not use ambient netrc credentials or proxy settings. Redirect destination restrictions and download byte/deadline limits remain separate hardening requirements. Dataset ZIP handling is unchanged.

Every path passed to the training loader ends in the exact lowercase suffix `.safetensors`. Valid extensionless, uppercase-suffix and `.sft` downloads are renamed locally before cache publication. This also applies when no network volume exists or a volume write fails, so the fallback cannot select a pickle loader merely because the download lacks a suffix.

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

November 12, 2026 verification: uv 0.13.8 resolved the complete registry requirement set for Linux amd64 and Python 3.10. The editable fork was omitted from this metadata-only check because its setup declares no dependencies. No fork clone, image build, GPU run or deployment was performed. The constrained HTTP subset also passed Python 3.11 synthetic loopback checks for requests model streaming, HTTP error cleanup, aiohttp/httpx transport, Jinja2 escaping and yarl URL handling. This does not verify Linux native wheels or GPU behavior. Image installation and runtime behavior still require acceptance.

For first adoption of these pins, use the deployment and safetensors transition checks in this document. Confirm installed package versions in the built image, scan its resolved Python and native packages, and verify downloads, startup, finite-loss training, callbacks and output before completing adoption. Subsequent deployments retaining these versions need no separate configuration step. Retain a verified compatible image for forward recovery. Do not use a pickle-accepting image as routine rollback.

The coordinated runtime upgrade is prepared locally as described next. Image/GPU acceptance and remaining dependency advisories are still open.

## Coordinated Python and GPU runtime upgrade

This candidate uses Python 3.11 in `/opt/venv`, PyTorch 2.15.0, torchvision 0.29.2, Pillow 13.0.0, Triton 3.9.0, bitsandbytes 0.50.2, filelock 4.2.10 and SymPy 1.14.0. NumPy 1.25.2 and the training-framework pins remain unchanged to limit compatibility changes. Both sd-scripts pins still select `59e616d1d1b5c52f2984d7004a96c2cff4b6ace3`. No training algorithm, job schema or callback change is included.

The NVIDIA base is CUDA 13.2.1 on Ubuntu 22.04, pinned by manifest digest. PyTorch's wheels supply CUDA 13.2.2 libraries. xformers 0.0.35 uses the explicit SHA-256-pinned CUDA 13.0 wheel from the official PyTorch index, not PyPI's CUDA 12.8 wheel. Its stable PyTorch ABI supports 2.10 and later. Native CUDA compatibility still requires execution on the intended GPU/driver. bitsandbytes includes its CUDA 13.2 backend. Do not mix the default CUDA 12 xformers binary into this image.

The headless training path does not require Tk. Published Python 3.11, development and venv packages were verified for Ubuntu 22.04. The venv PATH applies to pip, python3, accelerate, tokenizer caching and training subprocesses. The build checks the interpreter, runs `pip check`, and executes CPU compatibility checks against the actual pinned training code. It no longer force-reinstalls NumPy after resolving dependencies. The manifest remains authoritative.

### Local evidence and limits

November 12, 2026: registry resolution passed for Linux amd64/Python 3.11 with the selected GPU wheel. The NVIDIA manifest was verified to include amd64. CPU compatibility checks passed on both the previous torch 2.3.1/torchvision 0.18.1/Pillow 11.1.0 pair and the candidate under Python 3.11. They exercise the SDXL entry point, CLI contract, supported image formats and modes, resizing/cropping, gradient checkpointing, attention, tiny CLIP/VAE forward/backward passes, LoRA optimization and safetensors output. These are synthetic component checks, not a full SDXL training run or an output-quality guarantee. The 27 worker tests also pass. GPU checks are skipped on CPU and explicitly required by the release command. No image build, live GPU run or deployment is verified.

### Linux build attempt

November 12, 2026: the Linux amd64 build resolved and unpacked the pinned CUDA base successfully. It stopped during system package upgrade with apt exit code 100 because Docker's internal filesystem could not accommodate 843 MB of package archives. Python dependency installation, build-time compatibility checks and tokenizer caching were not reached. This is a storage blocker, not evidence that the candidate passes or fails Linux runtime compatibility. Restore sufficient Docker and host disk headroom, then rerun the full build without bypassing package updates or checks. Existing images, containers and volumes must not be deleted without operator authorization.

The 27 worker regression tests and shell syntax checks still pass locally. A complete image, offline tokenizer/handler checks and native GPU acceptance remain outstanding.

### One-time adoption

Applies when moving from Python 3.10/CUDA 12.1/torch 2.3.1 to this candidate. A production-connected push can automatically adopt the new image, so complete isolated acceptance before that release. An approved isolated NVIDIA GPU host is a prerequisite. If none is available, the upgrade is not ready for production.

1. Build the candidate image for linux/amd64 through an authorized build process. Confirm `pip check`, CPU compatibility checks and tokenizer caching succeed. Record the image digest and installed package versions, and scan the image's Python and native libraries.
2. On the approved isolated GPU host, run the following against the built image without production secrets or volumes:

   ```bash
   docker run --rm --gpus all runpod-worker-kohya:local \
     python3 /checks/runtime_compatibility.py --sd-scripts-root /sd-scripts --require-cuda
   ```

   This requires CUDA, checks the torch CUDA version and GPU compute capability, and executes xformers attention/backpropagation plus a two-step AdamW8bit update. It must not be reported as passing when GPU tests are skipped.
3. Before production adoption, verify endpoint capacity for CUDA 13.2 or newer and the intended GPU families. The candidate's native checks require compute capability 7.5 or later. Change the endpoint's minimum CUDA selection from 12.1 to 13.2 before the candidate serves jobs. Do not change production settings during local verification. Confirm intended GPU availability instead of assuming all previous hosts remain eligible.
4. Run an approved bounded private training acceptance job on the candidate. Compare finite loss, output usability, peak memory, runtime, callbacks and refunds with an approved baseline. Check both supported precision paths on appropriate GPUs, cache behavior, dataset loading and private output. The synthetic component suite does not substitute for this check.
5. After acceptance and an authorized release, confirm the intended image is active across every claiming worker. Preserve the format/cache and private-output safeguards during recovery. Do not routinely roll back to the pickle-accepting image. Keep legacy caches until separately approved cleanup. Record unresolved advisories separately.

No app payload migration or secret change is required. For provider commands needing existing credentials, load env vars through the approved process (see `.env.example`). This transition ends after image, GPU, training and all-worker checks pass. Later deployments retaining the same runtime contract require no repeat CUDA-setting change.

Sources: [PyTorch package metadata](https://pypi.org/pypi/torch/2.15.0/json), [xformers stable ABI](https://github.com/facebookresearch/xformers/releases/tag/v0.0.34), [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation), [Pillow 13 changes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html).
