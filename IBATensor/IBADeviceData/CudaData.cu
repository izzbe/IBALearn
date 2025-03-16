#include "CudaData.cuh"
#include <iostream>
// ------------------------------------------------ Constructors ------------------------------------------------------
CudaData::CudaData(size_t size) : size(size) {
      cudaMalloc(&head, size * sizeof(float));
}

CudaData::CudaData(const std::vector<float> &data_to_copy) : size(data_to_copy.size()) {
      cudaMalloc(&head, size * sizeof(float));
      cudaMemcpy(head, data_to_copy.data(),
            size * sizeof(float), cudaMemcpyHostToDevice);
}

CudaData::CudaData(float *C, size_t size) : head(C), size(size) {}

// -------------------------------------------------- Big 5 ----------------------------------------------------------
CudaData::CudaData(const DeviceData &other) : size(other.getSize()), head(nullptr) {
      cudaMalloc(&head, size * sizeof(float));
      cudaMemcpy(head, other.getData(), size * sizeof(float), cudaMemcpyHostToDevice);
}

CudaData &CudaData::operator=(const DeviceData &other) {
      if (this == &other) { return *this; };
      CudaData temp(other);
      std::swap(head, temp.head);
      return *this;
}

CudaData::CudaData(DeviceData &&other) noexcept {
      head = other.getData();
      size = other.getSize();
      other.getData() = nullptr;
      other.getSize() = 0;

}

CudaData &CudaData::operator=(DeviceData &&other) noexcept {
      if (this == &other) { return *this; };
      std::swap(head, other.getData());
      std::swap(size, other.getSize());
      return *this;
}

CudaData::~CudaData() {
     if (head) cudaFree(head);
}

// -------------------------------------------------- Getters --------------------------------------------------------

float *CudaData::getData() const { return head; }

size_t CudaData::getSize() const { return size; }

float *&CudaData::getData() { return head; }

size_t &CudaData::getSize() { return size; }

float CudaData::iloc(int i) const {
      if (i < 0 || i >= size) {
            throw std::logic_error("Iloc index out of bounds");
      }
      float value;
      cudaMemcpy(&value, head + i, sizeof(float), cudaMemcpyDeviceToHost);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
            std::cerr << "CUDA Memcpy Error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA iloc memory copy failed.");
      }
      return value;
}

// ------------------------------------------------- Matrix Ops -------------------------------------------------------
std::unique_ptr<DeviceData> CudaData::elemAdd(const DeviceData *other) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Add);
}

std::unique_ptr<DeviceData> CudaData::elemSub(const DeviceData *other) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Sub);
}

std::unique_ptr<DeviceData> CudaData::elemMult(const DeviceData *other) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Mult);
}

std::unique_ptr<DeviceData> CudaData::mat_mult(const DeviceData *other, int m, int k, int n) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return mat_mult_base(this, type_check, m, k, n);
}

__global__ void printRecursiveKernel(float *head, int *shape, int dims, int index, int stride, int depth);


// ------------------------------------------------ Kernals ------------------------------------------------------

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


// -------------------------------------------------- Helpers --------------------------------------------------------

__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B, int m, int k, int n) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim ((n+TILE_SIZE - 1) / TILE_SIZE, (m+TILE_SIZE -1) / TILE_SIZE);
      float *C;
      int size = m * n * sizeof(float);

      cudaMalloc(&C, size);

      mat_mult_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, m, k, n);

      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(C,m * n);
}

__host__ std::unique_ptr<DeviceData> elem_wise(const CudaData *A, const CudaData *B, Operation o) {
      dim3 blockDim(TILE_SIZE * TILE_SIZE);
      dim3 gridDim((A->getSize() + (TILE_SIZE*TILE_SIZE) - 1) / (TILE_SIZE * TILE_SIZE));
      float *C;
      int size = A->getSize() * sizeof(float);

      cudaMalloc(&C, size);

      if (o == Operation::Add) {
            mat_add_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize());
      } else if (o == Operation::Sub) {
            mat_sub_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize());
      } else if (o == Operation::Mult) {
            mat_element_mult_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize());
      }

      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(C, A->getSize());

}