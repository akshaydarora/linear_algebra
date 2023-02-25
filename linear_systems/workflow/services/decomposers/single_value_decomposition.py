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
    
    def getSVDLeastSquares(self,decomposed,B):
        U=decomposed["U"]
        VT=decomposed["VT"]
        sigma=decomposed["sigma"]
        # Moore–Penrose Pseudoinverse Method
        Sigma_pinv = np.zeros((self.n,self.m)).T
        Sigma_pinv[:self.m,:self.m] = np.diag(1/sigma[:self.m])
        x_lcs = VT.T.dot(Sigma_pinv).dot(U.T).dot(B)
        return x_lcs
    
    def getSVDLeastSquaresNorm(self,decomposed,B):
        x_lcs=self.getSVDLeastSquares(decomposed,B)
        # Compute ||AX-B||
        LS_norm=linalg.norm(self.A.dot(x_lcs)-B, 2)
        x_lcs_norm=linalg.norm(x_lcs)
        return LS_norm,x_lcs_norm


        
            