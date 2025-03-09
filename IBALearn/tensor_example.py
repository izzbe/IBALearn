import numpy as np
import ibatensor

def main():
    # Create a tensor from shape
    tensor1 = ibatensor.Tensor([2, 3])
    print("Empty tensor created with shape [2, 3]:")
    # tensor1.print()
    
    # Create a tensor from NumPy array
    # np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    # tensor2 = ibatensor.Tensor(np_array)
    # print("\nTensor created from NumPy array:")
    # tensor2.print()
    
    # # Get properties
    # print("\nProperties of the tensor:")
    # print(f"Shape: {tensor2.shape()}")
    # print(f"Strides: {tensor2.strides()}")
    # print(f"Size: {tensor2.size()}")
    # print(f"Dimensions: {tensor2.ndim()}")
    
    # # Access elements
    # print("\nAccessing elements:")
    # print(f"Element at [0, 0]: {tensor2.at([0, 0])}")
    # print(f"Element at [1, 2]: {tensor2.at([1, 2])}")
    
    # # String representation
    # print("\nString representation:")
    # print(repr(tensor2))

if __name__ == "__main__":
    main()