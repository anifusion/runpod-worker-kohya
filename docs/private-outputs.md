# Private model outputs

`private_output` is an optional boolean input, defaulting to false for legacy jobs.
When true, the worker requires `R2_PRIVATE_BUCKET` and the existing bucket credentials,
writes `private-v1/lora/<out_id>.safetensors` without a public ACL, and returns a temporary
signed URL. Missing private configuration fails the job without a public or local fallback.
The output id must contain only letters, digits, hyphens, and underscores (128 characters at most) or the job fails before any download.
The bucket must have no public domain or anonymous read policy. Configure it before
submitting private jobs, and verify the completed build and a synthetic training result.
Keep workers supporting this input available while private jobs are queued or retryable.
Legacy jobs retain their original bucket and key. See `.env.example` for placeholder configuration.
