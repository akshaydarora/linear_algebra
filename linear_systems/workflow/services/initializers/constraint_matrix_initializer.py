import numpy as np

class Matrix(object):
    def __new__(cls, *args, **kwargs):
        print(" Created a new instance of Matrix:A")
        return super().__new__(cls)

    def __init__(self, A):
        print(" Initialized a new instance of Matrix :A")
        self.A = np.array(A)
        self.metadata={
            "matrix_size":self.A.shape,
            "matrix_rows":self.A.shape[0],
            "matrix_cols":self.A.shape[1],
            "matrix_dim":self.A.ndim
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metadata={self.metadata})"