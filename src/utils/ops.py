import torch
import theseus as th
from torch import Tensor
from theseus.geometry import SO3


def add(y, x):
    if isinstance(x, SO3):
        # return y @ x  # Matrix multiplication if x is an SO3 object
        return y.compose(x)
    else:
        return y + x  # Element-wise addition for non-SO3 types

def main():
    matrix = SO3.rand(1)  # Generates a SO3 instance of shape (1, 3, 3)
    matrix2 = SO3.rand(1)  # Generates a SO3 instance of shape (1, 3, 3)
    # print(matrix)
    # print(matrix2)

    m = add(matrix, matrix2)  # Will not work, mismatched shapes
    # print(m)
    
if __name__ == "__main__":
	main()