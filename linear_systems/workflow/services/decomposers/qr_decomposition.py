import numpy as np
from numpy import linalg

class QRDecomposition(object):

    def __init__(self,a,method):
        self.A = a.A
        self.method=method
        self.metadata=a.metadata
        self.n=self.metadata["matrix_rows"]
        self.m=self.metadata["matrix_cols"]
        self.rank=linalg.matrix_rank(self.A)
        print("rank:{}".format(self.rank))
        print("m:{}".format(self.m))

    def getQRDecomposeResult(self):
        # Gram-Schmidt Method
        if self.rank<self.m:
            raise Exception("rank is smaller than columns")
        Q= np.zeros((self.n,self.m))
        for i, col in enumerate(self.A.T):
            Q[:,i]=col
            for prev in Q.T[:i]:
                Q[:,i]-=(prev@col)/(prev@prev)*prev
        Q=Q/(linalg.norm(Q,axis=0))
        R=Q.T@self.A
        return Q,R
    
    def getQRLeastSquares(self,decomposed,B):
        Q=decomposed["Q"]
        R=decomposed["R"]
        QTB=Q.T@B
        
        x_lcs=np.linalg.solve(R, QTB)
        return x_lcs
    
    def getQRLeastSquaresNorm(self,decomposed,B):
        x_lcs=self.getQRLeastSquares(decomposed,B)
        # Compute ||AX-B||
        LS_norm=linalg.norm(self.A.dot(x_lcs)-B, 2)
        x_lcs_norm=linalg.norm(x_lcs)
        return LS_norm,x_lcs_norm
            