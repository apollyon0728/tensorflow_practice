from ADTest import svdRec
from numpy import *

data = svdRec.loadExData()
U, Sigma, VT = linalg.svd(data)
print(U)
print("=========================")
print(Sigma)
print("=========================")
print(VT)
