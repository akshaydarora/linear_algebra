import numpy as np
from numpy import linalg
from helper_utils.helpers import csqrt

class SVD(object):

    def __init__(self,a):
        self.A = a.A
        self.metadata=a.metadata
        self.n=self.metadata["matrix_rows"]
        self.m=self.metadata["matrix_cols"]
        self.rank=linalg.matrix_rank(self.A)

    def getSVDResult(self):

        if self.rank<self.m:
            raise Exception("rank is smaller than columns")
        _, u = linalg.eigh(np.linalg.inv(self.A@self.A.T))
        sigma,_=linalg.eigh(self.A@self.A.T)
        sigma=np.array([csqrt(i) for i in sigma])[::-1]
        sigma = abs(sigma)
        _,vt = linalg.eigh(np.linalg.inv(np.transpose(self.A)@self.A))
        return u,sigma,vt
            