from services.decomposers.qr_decomposition import QRDecomposition
from services.decomposers.single_value_decomposition import SVD
from services.factorizers.cholesky_factorizer import CholeskyFactorizer
from services.initializers.constraint_matrix_initializer import Matrix
from services.initializers.right_hand_side_initializer import RHSCheck

if __name__=="__main__":
    #Parameters
    A=[
            [4,-2,2],
            [-2,2,4],
            [2,-4,11]
        ]
    B=[1,2,3]
    qr_method="gram_schmidt"

    # Initialize the matrix A
    a=Matrix(A)
    b=RHSCheck(B)
    b.checkRHS()

    # Initialize Decomposition
    qr=QRDecomposition(a,qr_method)
    Q,R=qr.getQRDecomposeResult()
    
    svd=SVD(a)
    U,sigma,VT=svd.getSVDResult()
    
    # Initialize Factorization
    chs=CholeskyFactorizer(a)
    L=chs.getCHFResult()





