import json
import os
from services.initializers.constraint_matrix_initializer import Matrix
from linear_system import LinearSystem

if __name__=="__main__":

    dir_name=os.path.dirname(os.path.realpath(__file__)) 
    data_dir="files/input/"
    ######## Initialize Params #############
    with open(os.path.join(dir_name,"config/config.json"),"r",encoding='utf-8') as f:
        config=json.load(f)
    with open(os.path.join(dir_name,"config/input_config.json"),"r",encoding='utf-8') as f:
        input_config=json.load(f)
    with open(data_dir+"data_A.json","r",encoding='utf-8') as f:
        input_data=json.load(f)

    A=input_data["A"]    

    # Initialize the matrix A

    a=Matrix(A)

    ############ Create Linear System #####################

    # Initialize the Methods
    method_type=input_config["method_type"]
    method_name=input_config["method_name"]
    cls=LinearSystem(a,config)
    if method_type=="decomposition":
        decompose_status=cls.decompose(method_type,method_name)
        print(decompose_status)

    elif method_type=="factorization":
        factorize_status=cls.factorize(method_type,method_name)
        print(factorize_status)
