import json
import os
from services.initializers.right_hand_side_initializer import RHS
from services.save_load_ls.save_load_linear_system import SaveLoadLS
from services.decomposers.single_value_decomposition import SVD
from services.decomposers.qr_decomposition import QRDecomposition
from services.factorizers.cholesky_factorizer import CholeskyFactorizer
from services.initializers.constraint_matrix_initializer import Matrix

if __name__=="__main__":

    ######## Initialize Params #############
    dir_name=os.path.dirname(os.path.realpath(__file__)) 
    data_dir="files/input/"
    with open(os.path.join(dir_name,"config/config.json"),"r",encoding='utf-8') as f:
        config=json.load(f)
    with open(data_dir+"data_A.json","r",encoding='utf-8') as f:
        input_data=json.load(f)
    with open(data_dir+"data_B.json","r",encoding='utf-8') as f:
        rhs_data=json.load(f)

    ###### Load the Original Input A dataset ######
    A=input_data["A"]
    a=Matrix(A) 
    ###### Load the Original RHS B dataset ######
    B=rhs_data["B"]
    # Initialize the matrix B and check for acceptance
   
    b=RHS(B)
    b.checkRHS()

    ############ Solve/Compute Least Square Errors #####################

    #####Initialize the Methods#######

    method_type="factorization"
    method_name="cholesky"

    ##### LOAD THE DECOMPOSED/FACTORIZED Dataset ####
    file_dir=config[method_type][method_name]["file_dir"]
    file_name=config[method_type][method_name]["file_name"]
    sls=SaveLoadLS(file_obj=None,file_name=file_name,file_dir=file_dir)

    if method_type=="decomposition":
        if method_name=="svd":
            decomposed=sls.load_file()
            svd=SVD(a)
            x_lcs=svd.getSVDLeastSquares(decomposed=decomposed,B=B)
            LS_norm,x_lcs_norm=svd.getSVDLeastSquaresNorm(decomposed=decomposed,B=B)
            print(LS_norm,x_lcs_norm)
        elif method_name=="qr":
            decomposed=sls.load_file()
            qr=QRDecomposition(a,'gram_schmidt')
            x_lcs=qr.getQRLeastSquares(decomposed=decomposed,B=B)
            LS_norm,x_lcs_norm=qr.getQRLeastSquaresNorm(decomposed=decomposed,B=B)
            print(LS_norm,x_lcs_norm)

    elif method_type=="factorization":
        if method_name=="cholesky":
            factorized=sls.load_file()
            chf=CholeskyFactorizer(a)
            x_lcs=chf.getCHFLeastSquares(factorized=factorized,B=B)
            LS_norm,x_lcs_norm=chf.getCHFLeastSquaresNorm(factorized=factorized,B=B)
            print(LS_norm,x_lcs_norm)
    else:
        print("invalid method type recvd, please choose decomposition or factorization")    
