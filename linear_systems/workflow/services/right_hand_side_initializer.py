import numpy as np

class RHSCheck(object):

    def __init__(self, b):
        print(" Initialize a new instance of Matrix :B")
        self.b = np.array(b)
        self.metadata={
            "matrix_size":b.shape,
            "matrix_rows":b.shape[0],
            "matrix_dim":b.ndim
        }
    def __repr__(self) -> str:
        return f"{type(self).__name__}(metadata={self.metadata})"
    
    def checkRHS(self):
        assert self.metadata["matrix_dim"]==1