"""
security/hashing.py
====================
HMAC-SHA256 integrity hashing for model parameters.

Uses a shared secret key (FL_HASH_SECRET) from the environment
to produce a keyed hash. This prevents a rogue client from
crafting parameters that happen to match a known hash value.

If FL_HASH_SECRET is not set, a default fallback is used —
set it in your .env file for production use.
"""

import hashlib
import hmac
import os

import numpy as np

# Load secret from env; warn if using the insecure default
_SECRET = os.environ.get("FL_HASH_SECRET", "fl-default-secret-change-me").encode("utf-8")


def hash_parameters(parameters) -> str:
    """
    Compute a deterministic HMAC-SHA256 digest over all model parameter
    tensors. Includes each tensor's shape, dtype, and raw bytes.

    Args:
        parameters: Iterable of NumPy arrays (model weights).

    Returns:
        Lowercase hex string (64 characters).
    """
    mac = hmac.new(_SECRET, digestmod=hashlib.sha256)

    for param in parameters:
        param = np.ascontiguousarray(param)
        mac.update(str(param.shape).encode("utf-8"))
        mac.update(str(param.dtype).encode("utf-8"))
        mac.update(param.tobytes())

    return mac.hexdigest()