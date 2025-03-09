import numpy as np
from IBALearn.neural_network import ibatensor

def main():
    # Create a tensor from shape
    tensor1 = ibatensor.Tensor([2, 3])
    print("Empty tensor created with shape [2, 3]:")
    # tensor1.print()
    
    # Create a tensor from NumPy array
    # np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # # Get properties
    # print("\nProperties of the tensor:")
    # print(f"Shape: {tensor2.shape()}")
    # print(f"Strides: {tensor2.strides()}")
    # print(f"Size: {tensor2.size()}")
    # print(f"Dimensions: {tensor2.ndim()}")

    # # String representation
    # print("\nString representation:")
    # print(repr(tensor2))

if __name__ == "__main__":
    main()