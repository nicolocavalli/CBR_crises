# patch_pygam.py
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

# Fix deprecated np.int
if not hasattr(np, 'int'):
    np.int = int

# Restore .A attribute on csr_matrix and csc_matrix
for sparse_type in [csr_matrix, csc_matrix]:
    if not hasattr(sparse_type, 'A'):
        sparse_type.A = property(lambda self: self.toarray())
