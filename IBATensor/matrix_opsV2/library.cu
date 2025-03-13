#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "library.cuh"

enum class Operation { Add, Sub, Mult };


Data::Data(size_t size) : size(size) {
      cudaMalloc(&head, size * sizeof(float));
}

Data::Data(const std::vector<float> &data_to_copy, size_t size) : size(size) {
      cudaMalloc(&head, size * sizeof(float));
      cudaMemcpy(head, data_to_copy.data(),
            size * sizeof(float), cudaMemcpyHostToDevice);
}

Data::Data(float *C, size_t size) : head(C), size(size){}

Data::Data(const Data &other) : size(other.size) {
      cudaMalloc(&head, size * sizeof(float));
      cudaMemcpy(head, other.head, size * sizeof(float), cudaMemcpyHostToDevice);
}

Data &Data::operator=(const Data &other) {
      if (this == &other) { return *this; };
      Data temp(other);
      std::swap(head, temp.head);
      return *this;
}

Data::Data(Data &&other) noexcept {
      head = other.head;
      size = other.size;
      other.head = nullptr;
      other.size = 0;

}

Data &Data::operator=(Data &&other) noexcept {
      if (this == &other) { return *this; };
      std::swap(head, other.head);
      std::swap(size, other.size);
      return *this;
}

Data::~Data() {
     if (head) cudaFree(head);
}

float *Data::getData() const { return head; };

size_t Data::getSize() const { return size; };


__global__ void mat_mult_kernel(float A[], float B[], float C[], int m, int k, int n) {
      int global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
      int global_col = blockIdx.x * TILE_SIZE + threadIdx.x;

      __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
      __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

      float c_val = 0.0f;

      for (int i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; ++i) {
            if (global_row < m && (i * TILE_SIZE + threadIdx.x) < k) {
                  shared_A[threadIdx.y][threadIdx.x] = A[global_row * k + (i * TILE_SIZE + threadIdx.x)];
            } else {
                  shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (global_col < n && (i * TILE_SIZE + threadIdx.y) < k) {
                  shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + global_col];
            } else {
                  shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (int j = 0; j < TILE_SIZE; ++j) {
                  c_val += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
            }

            __syncthreads();
      }

      if (global_row < m && global_col < n) {
            C[global_row * n + global_col] = c_val;
      }
}

__global__ void mat_add_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] + B[global_ind % b_size];
      }
}

__global__ void mat_sub_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] - B[global_ind % b_size];
      }
}

__global__ void mat_element_mult_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] * B[global_ind % b_size];
      }
}

__host__ Data mat_mult(Data A, Data B, int m, int k, int n) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim ((n+TILE_SIZE - 1) / TILE_SIZE, (m+TILE_SIZE -1) / TILE_SIZE);
      float *C;
      int size = m * n * sizeof(float);

      cudaMalloc(&C, size);

      mat_mult_kernel<<<gridDim, blockDim>>>(A.getData(), B.getData(), C, m, k, n);

      cudaDeviceSynchronize();

      return Data(C, m * n);
}

__host__ Data elem_wise(Data A, Data B, Operation o) {
      dim3 blockDim(TILE_SIZE * TILE_SIZE);
      dim3 gridDim((A.getSize() + (TILE_SIZE*TILE_SIZE) - 1) / (TILE_SIZE * TILE_SIZE));
      float *C;
      int size = A.getSize() * sizeof(float);

      cudaMalloc(&C, size);

      if (o == Operation::Add) {
            mat_add_kernel<<<gridDim, blockDim>>>(A.getData(), B.getData(), C, A.getSize(), B.getSize(), A.getSize());
      } else if (o == Operation::Sub) {
            mat_sub_kernel<<<gridDim, blockDim>>>(A.getData(), B.getData(), C, A.getSize(), B.getSize(), A.getSize());
      } else if (o == Operation::Mult) {
            mat_sub_kernel<<<gridDim, blockDim>>>(A.getData(), B.getData(), C, A.getSize(), B.getSize(), A.getSize());
      }

      cudaDeviceSynchronize();

      return Data(C, A.getSize());
}
