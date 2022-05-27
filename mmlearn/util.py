import logging

from torch import cuda
from torch import device as ptdevice

# Training param defaults
REG_PARAM_RANGE = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
MAX_ITER_SKLEARN = 100000

# Paths
SAVE_DIR = "pers"
DATA_DIR = "data"
RESULTS_DIR = "results"

#DEVICE = ptdevice("cuda" if cuda.is_available() else "cpu")
DEVICE = ptdevice("cpu")
#USE_CUDA = cuda.is_available()
USE_CUDA = False

# def device():
#     return ptdevice("cuda" if cuda.is_available() else "cpu")

# def cuda_available():
#     return cuda.is_available()

def log_progress(msg, color="yellow"):
    """Log activity messages to INFO logger"""
    
    colors = {
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[32m",
        "white": "\u001b[0m"
    }
    logging.info(f"{colors[color]}{msg}\u001b[0m")