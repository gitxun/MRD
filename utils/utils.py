import os
import random
import numpy as np
import torch

# seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ensure_dir_exists(dir_path):
    """
    Ensure that the specified directory exists. If not, create it.

    :param dir_path: Path to the directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)