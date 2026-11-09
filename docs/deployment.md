# Deployment and checkpoint cache

## Deployment

Runpod can build from a connected GitHub repository automatically after a push.

For a local build, use `docker build --platform linux/amd64 -t runpod-worker-kohya:local .`. The pinned Triton version does not provide a compatible arm64 wheel.

**Manual action after each push:** Open the endpoint's build history in Runpod. Confirm that the build for the pushed commit is marked `Completed` and the endpoint uses that image before running jobs.

## Checkpoint cache recovery

The worker checks a cached checkpoint's size, header, and filename suffix before reuse. An invalid cache is ignored until a validated legacy copy or download replaces its path. Legacy extensionless caches are copied to a filename matching their detected format. Failed migration falls back to the normal download path without selecting the extensionless file.

Downloads are checked before training, even without a network volume. When a volume is available, the worker stages and validates a copy there before publishing it atomically. A cache write failure leaves the existing cache intact and trains from the validated download. Header checks catch obvious wrong-format and some partial files, but they do not verify a full checkpoint against an expected hash.

Run synthetic worker regression checks with Python 3.10 or later: `python3 -B -m unittest discover -s tests`. These tests do not download models or run GPU training.
