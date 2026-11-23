                   
import hashlib
import random

import numpy as np


def derive_seed(*parts):
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)


def configure_torch(threads):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
