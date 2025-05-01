# patch_pygam.py
import numpy as np
from scipy.sparse import csr_matrix

# Fix deprecated np.int
if not hasattr(np, 'int'):
    np.int = int

# Restore .A attribute for scipy csr_matrix
if not hasattr(csr_matrix, 'A'):
    csr_matrix.A = property(lambda self: self.toarray())
