"""Unmix backend: neural audio source separation behind a small FastAPI service."""

import os as _os

# Privacy defaults, set before any library that reads them is imported. Model
# downloads still contact their hosts (GitHub, Meta's CDN, Hugging Face), but
# nothing else: no usage telemetry, and no stored Hugging Face token is sent.
for _key, _value in {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
}.items():
    _os.environ.setdefault(_key, _value)
