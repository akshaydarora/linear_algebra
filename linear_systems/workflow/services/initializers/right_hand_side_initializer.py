import numpy as np

class RHSCheck(object):

    def __init__(self, b):
        print("Initialized the new instance of b")
        self.b = np.array(b)
        self.metadata={
            "matrix_size":self.b.shape,
            "matrix_rows":self.b.shape[0],
            "matrix_dim":self.b.ndim
        }
    def __repr__(self) -> str:
        return f"{type(self).__name__}(metadata={self.metadata})"
    
    def checkRHS(self):
        assert self.metadata["matrix_dim"]==1, "invalid input, needs 1-dimensional matrix B"