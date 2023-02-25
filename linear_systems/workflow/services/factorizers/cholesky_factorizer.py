import numpy as np
from numpy import linalg
from helper_utils.helpers import csqrt,isposdef


class CholeskyFactorizer(object):

    def __init__(self,a):
        self.A = a.A
        self.metadata=a.metadata
        self.n=self.metadata["matrix_rows"]
        self.m=self.metadata["matrix_cols"]
        self.rank=linalg.matrix_rank(self.A)

    def test_positive_definite(self):
        assert isposdef(self.A)==True,"matrix is not positive definite"

    def getCHFResult(self):
        self.test_positive_definite()
        l=self.A.copy()
        n=len(l)
        if self.rank<self.m:
            raise Exception("rank is smaller than columns")
        for k in range(n):
            # diagonal elements
            l[k,k]=round(csqrt(l[k,k] - np.dot(l[k,0:k],l[k,0:k])),2)        
            # non-diagonal elements
            for i in range(k+1,n):
                l[i,k]=(l[i,k]-np.dot(l[i,0:k],l[k,0:k]))/l[k,k]
        for k in range(1,n):
            l[0:k,k]=0.0
        return l 