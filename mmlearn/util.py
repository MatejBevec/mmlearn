import logging, sys, io

from torch import cuda
from torch import device as ptdevice

# Training param defaults
REG_PARAM_RANGE = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
MAX_ITER_SKLEARN = 100000
SAMPLE_RATE = 16000

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

def log_progress(msg, color="yellow", level="info", verbose=False):
    """Log activity messages to INFO logger"""
    
    colors = {
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[32m",
        "white": "\u001b[0m"
    }
    levels = {"error": 40, "warning": 30, "info": 20, "debug": 10}

    msg = f"{colors[color]}{msg}\u001b[0m"
    logging.log(levels[level], msg)
    if verbose and logging.root.level > levels[level]:
        print(msg) # Not sure if this is the correct way


def check_dict_like(input, return_list=False):
    """Check if input is a dict, a list of values or a list of ("key", value) tuples.
        Convert to dict or list and return."""
    
    pass

def silence_call():
    pass
