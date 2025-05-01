# patch_pygam.py
import numpy as np
import pygam

# Monkey-patch deprecated np.int to builtin int
if not hasattr(np, 'int'):
    np.int = int
