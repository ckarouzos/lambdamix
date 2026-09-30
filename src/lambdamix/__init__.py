"""Preference-conditioned low-rank training and evaluation."""

__version__ = "0.1.0"


def resolve_attn_implementation(requested: str | None = None) -> str:
    """Use PyTorch attention unless another backend is explicitly requested."""
    return requested or "sdpa"
