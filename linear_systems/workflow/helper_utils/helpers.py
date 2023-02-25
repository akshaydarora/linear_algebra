from ctypes import c_float, c_long, byref, POINTER, cast
import numpy as np

def csqrt(num):

    xhalf = 0.5*num
    x = c_float(num)
    i = cast(byref(x), POINTER(c_long)).contents.value
    i = c_long(0x5f375a86 - (i>>1))
    x = cast(byref(i), POINTER(c_float)).contents.value
    x = x*(1.5-xhalf*x*x)
    x = x*(1.5-xhalf*x*x)
    sqrt_res=x*num
    return sqrt_res

def isposdef(x):

    status= np.all(np.linalg.eigvals(x)>0)
    return status
