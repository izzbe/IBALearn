import numpy as np
from IBALearn.neural_network import ibatensor

def main():
    # Create a tensor from shape
    tensor1 = ibatensor.Tensor([2, 3])
    print("Empty tensor created with shape [2, 3]:")
    tensor1.print()
    
    # Create a tensor from NumPy array
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    tensor2 = ibatensor.Tensor(np_array)
    tensor3 = ibatensor.Tensor(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))

    # Get properties
    tensor2.print()
    print(f"Shape: {tensor2.shape}")
    print(f"Strides: {tensor2.stride}")
    print(f"Size: {tensor2.size}")

    tensor3.print()
    tensor4 = tensor2 @ tensor3
    tensor4.print()

    tensor5 = ibatensor.Tensor(np.array([1.0], dtype=np.float32))

    (tensor2 @ tensor5).print()

    # # String representation
    # print("\nString representation:")
    # print(repr(tensor2))

if __name__ == "__main__":
    main()