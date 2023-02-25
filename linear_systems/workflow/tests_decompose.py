import unittest
from services.decomposers.qr_decomposition import QRDecomposition
from services.decomposers.single_value_decomposition import SVD
from services.factorizers.cholesky_factorizer import CholeskyFactorizer
from services.initializers.constraint_matrix_initializer import Matrix
import numpy as np

class Tests(unittest.TestCase):

    def testDecompositionQR(self):
          A=[[25, 15, -5],
             [15, 18, 0],
             [-5, 0, 11]]
          a=Matrix(A)
          A_mat=a.A
          qr=QRDecomposition(a,method="gram_schmidt")
          Q,R=qr.getQRDecomposeResult()
          I=Q.T@Q
          self.assertEqual(np.allclose(Q@R,A_mat), True)
          self.assertEqual(np.allclose(I,np.eye(I.shape[0])),True) 

    def testDecomposition2D_SVD(self):

          A=[[3.44,1.121212],[4.2121311,2.2323232323121212121]]
          a=Matrix(A)
          A_mat=a.A
          svd=SVD(a)
          U,sigma,VT=svd.getSVDResult()
          self.assertEqual(np.allclose(U @ np.diag(sigma) @ VT,A_mat), True)

    def testFactorizationCHOLESKY(self):
          A=[[25, 15, -5],
             [15, 18, 0],
             [-5, 0, 11]]
          a=Matrix(A)
          A_mat=a.A
          chf=CholeskyFactorizer(a)
          L=chf.getCHFResult()
          print(L@L.T==A_mat)
          self.assertEqual(np.allclose(np.array(L@L.T),np.array(A_mat)), True)   


if __name__ == '__main__':

    unittest.main()    
