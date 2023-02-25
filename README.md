# linear_algebra

![parabola](https://user-images.githubusercontent.com/26202862/221382371-ef9a2135-6050-4818-b7c3-8b4038597c4b.png)


 
The Linear algebra module is a collection of various decomposition and factorization algorithms
used to solve for overdetermined homogeneous linear system

# Steps to Compute the Least Squares 

Step 1 : Run the create linear system to initialize the Matrix A and precompute the decomposition Matrix
    Configure the Decomposition/Factorization Method from one of the configurations below
    
    
    {
    "method_type":"decomposition",
    "method_name":"qr"
    }

    {
    "method_type":"decomposition",
    "method_name":"svd"
    }

    {
    "method_type":"factorization",
    "method_name":"cholesky"
    }
--> run $create_linear_system.py
--> output $ {'status': 'success', 'message': 'successully saved the qr : decomposed file qr in files/transformed/QR location'}

Step 2 : Solve the Least Squares for X using decomposition methods and get Least Squares Norm

--> run $solve_least_squares.py


