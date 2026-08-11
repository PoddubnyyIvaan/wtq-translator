from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> dict[str, bool]:
    """Seed available local RNGs without making NumPy/Torch hard dependencies."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    seeded = {"python": True, "numpy": False, "torch": False}

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    except ImportError:
        pass

    return seeded
