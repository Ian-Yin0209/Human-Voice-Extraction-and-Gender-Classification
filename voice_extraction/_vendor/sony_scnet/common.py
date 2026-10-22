"""Local equivalents of the two upstream helpers used by the backbone."""
from functools import partial
from torch.utils.checkpoint import checkpoint as torch_checkpoint

checkpoint = partial(torch_checkpoint, use_reentrant=False)


def exists(value):
    return value is not None
