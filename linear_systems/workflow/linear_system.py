from services.decomposers.qr_decomposition import QRDecomposition
from services.decomposers.single_value_decomposition import SVD
from services.factorizers.cholesky_factorizer import CholeskyFactorizer
from services.save_load_ls.save_load_linear_system import SaveLoadLS

class LinearSystem(object):

    def __init__(self,a,config):
        
        self.a=a
        self.config=config
    
    # Initialize Decomposition
    def decompose(self,method_type,method_name,*args,**kwargs):

        decomposition_status={}
        file_dir=self.config[method_type][method_name]["file_dir"]
        file_name=self.config[method_type][method_name]["file_name"]
        if method_name.upper()=="QR":
            qr_method="gram_schmidt"
            qr=QRDecomposition(self.a,qr_method)
            Q,R=qr.getQRDecomposeResult()
            QR_result={}
            QR_result["Q"]=Q
            QR_result["R"]=R
            sls=SaveLoadLS(file_obj=QR_result,file_name=file_name,file_dir=file_dir)
            sls.save_file()
            decomposition_status["status"]="success"
            decomposition_status["message"]="successully saved the {} : decomposed file{} in {} location".format(method_name,
                                                                                                                  file_name,
                                                                                                                  file_dir)


        elif method_name.upper()=="SVD":
            svd=SVD(self.a)
            U,sigma,VT=svd.getSVDResult()
            SVD_result={}
            SVD_result["U"]=U
            SVD_result["sigma"]=sigma
            SVD_result["VT"]=VT
            sls=SaveLoadLS(file_obj=SVD_result,file_name=file_name,file_dir=file_dir)
            sls.save_file()
            decomposition_status["status"]="success"
            decomposition_status["message"]="successully saved the {} : decomposed file {} in {} location".format(method_name,
                                                                                                                  file_name,
                                                                                                                  file_dir)
        else:
            raise Exception("please choose decompose method in ('QR','SVD')")

        return decomposition_status    
    
    # Initialize Factorization
    def factorize(self,method_type,method_name,*args,**kwargs):

        factorization_status={}
        file_dir=self.config[method_type][method_name]["file_dir"]
        file_name=self.config[method_type][method_name]["file_name"]
        if method_name.upper()=="CHOLESKY":
            L_result={}
            chs=CholeskyFactorizer(self.a)
            L=chs.getCHFResult()
            L_result["L"]=L
            sls=SaveLoadLS(file_obj=L_result,file_name=file_name,file_dir=file_dir)
            sls.save_file()
            factorization_status["status"]="success"
            factorization_status["message"]="successully saved the {} : factorized file {} in {} location".format(method_name,
                                                                                                                  file_name,
                                                                                                                  file_dir)
        else:
            raise Exception("please choose factorize method in ('CHS')") 
        
        return factorization_status
    