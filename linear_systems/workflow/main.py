import json
import os
from services.initializers.constraint_matrix_initializer import Matrix
from services.initializers.right_hand_side_initializer import RHS
from create_linear_system import CreateLinearSystem


if __name__=="__main__":
    dir_name=os.path.dirname(os.path.realpath(__file__)) 
    ######## Initialize Params #############
    with open(os.path.join(dir_name,"config/config.json"),"r",encoding='utf-8') as f:
        config=json.load(f)

    A=[
            [4,-2,2],
            [-2,2,4],
            [2,-4,11]
        ]
    B=[1,2,3]

    # Initialize the matrix A
    a=Matrix(A)

    b=RHS(B)
    b.checkRHS()

    ############ Create Linear System #####################

    # Initialize the Methods
    method_type="factorization"
    method_name="cholesky"
    cls=CreateLinearSystem(a,config)
    if method_type=="decomposition":
        decompose_status=cls.decompose(method_type,method_name)
        print(decompose_status)

    elif method_type=="factorization":
        factorize_status=cls.factorize(method_type,method_name)
        print(factorize_status)


