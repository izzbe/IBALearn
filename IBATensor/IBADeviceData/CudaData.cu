#include "CudaData.cuh"
#include <iostream>
#include <cfloat>
// ------------------------------------------------ Decls -------------------------------------------------------------
__host__ std::unique_ptr<DeviceData> elem_wise(const CudaData *A, const CudaData *B, Operation o);
__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B, int m, int k, int n);
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

__global__ void set_index_kernel(float A[], int i, float val);
void CudaData::set_index(int i, float val) {
	if(i < 0 || i > size) {
    	throw std::logic_error("CUDA set_index out of bound");
    }

    set_index_kernel<<<1, 1>>>(head, i, val);
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

std::unique_ptr<DeviceData> conv2d_base(const float *in, const float *kern, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S, int C_out);

std::unique_ptr<DeviceData> CudaData::conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(kern);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }
      return conv2d_base(head, type_check->getData(), N, C_in, H, W, H_out, W_out, K, P, S, C_out);
}

std::unique_ptr<DeviceData> pool_base(const float *in, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S, Pool option);

std::unique_ptr<DeviceData> CudaData::avg_pool(int N, int C_in, int H, int W, int H_out, int W_out,
                                               int K, int P, int S) const {
      return pool_base(head, N, C_in, H, W, H_out, W_out, K, P, S, Pool::Average);
}

std::unique_ptr<DeviceData> CudaData::max_pool(int N, int C_in, int H, int W, int H_out, int W_out,
                                               int K, int P, int S) const {
      return pool_base(head, N, C_in, H, W, H_out, W_out, K, P, S, Pool::Max);
}

std::unique_ptr<DeviceData> mat_transpose_base(const float *in, int H, int W, int C, int N);

std::unique_ptr<DeviceData> CudaData::mat_transpose(int H, int W, int C, int N) const {
      return mat_transpose_base(head, H, W, C, N);
}

std::unique_ptr<DeviceData> relu_base(const float *in, int H, int W, int C, int N);
std::unique_ptr<DeviceData> CudaData::relu(int H, int W, int C, int N) const {
      return relu_base(head, H, W, C, N);
}
// ------------------------------------------------ Kernels ------------------------------------------------------

__global__ void set_index_kernel(float A[], int i, float val) {
	A[i] = val;
}

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

__global__ void conv2d_kernel(const float *in, const float *kern, float *out, int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_out;
      int batch_n = blockIdx.z / C_out;

      if (out_x >= W_out || out_y >= H_out || out_c >= C_out || batch_n >= N) return;

      int input_x = out_x * S - P;
      int input_y = out_y * S - P;

      float val = 0.0f;
      for (int c = 0; c < C_in; ++c) {
            for (int i = 0; i < K; ++i) {
                  for (int j = 0; j < K; ++j) {
                        int read_x = input_x + j;
                        int read_y = input_y + i;
                        if (read_x >= 0 && read_x < W && read_y >= 0 && read_y < H) {
                              val += in[batch_n * C_in * H * W + c * H * W + read_y * W + read_x ] *
                              kern[out_c * C_in * K * K + c * K * K + i * K + j];
                        }
                  }
            }
      }
      out[batch_n * C_out * H_out * W_out + out_c * H_out * W_out + out_y * W_out + out_x] = val;

}

int K = 3;     // kernel
int S = 2;     // stride
int P = 1;

__global__ void max_pool2d_kernel(const float *in, float *out, int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_in;
      int batch_n = blockIdx.z / C_in;

      if (out_x >= W_out || out_y >= H_out || out_c >= C_in || batch_n >= N) return;

      int input_x = out_x * S - P;
      int input_y = out_y * S - P;

      float cur_max = -FLT_MAX;

      for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                  int read_x = input_x + j;
                  int read_y = input_y + i;
                  if (read_x >= 0 && read_x < W && read_y >= 0 && read_y < H) {
                        if (in[batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x] > cur_max) {
                              cur_max = in[batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x];
                        }
                  }
            }
      }

      out[batch_n * C_in * H_out * W_out + out_c * H_out * W_out + out_y * W_out + out_x] = cur_max;

}

__global__ void avg_pool2d_kernel(const float *in, float *out, int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_in;
      int batch_n = blockIdx.z / C_in;

      if (out_x >= W_out || out_y >= H_out || out_c >= C_in || batch_n >= N) return;

      int input_x = out_x * S - P;
      int input_y = out_y * S - P;

      float sum = 0;

      for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                  int read_x = input_x + j;
                  int read_y = input_y + i;
                  if (read_x >= 0 && read_x < W && read_y >= 0 && read_y < H) {
                        sum += in[batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x];
                  }
            }
      }

      out[batch_n * C_in * H_out * W_out + out_c * H_out * W_out + out_y * W_out + out_x] = sum / (K * K);

}

__global__ void mat_transpose_kernel(const float *in, float *out, int H, int W, int C, int N) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y +threadIdx.y;
      int out_c = blockIdx.z % C;
      int batch_n = blockIdx.z / C;

      if (out_x >= H || out_y >= W || out_c >= C || batch_n >= N) {
            return;
      }

      int in_x = out_y;
      int in_y = out_x;

      out[ batch_n * C * H * W + out_c * H * W + out_y * H + out_x ] = in[ batch_n * C * H * W + out_c * H * W + in_y * W + in_x ];
}

__global__ void relu_kernel(const float *in, float * out, int H, int W, int C, int N) {
      int x_ind = blockIdx.x * blockDim.x + threadIdx.x;
      int y_ind = blockIdx.y * blockDim.y + threadIdx.y;
      int c_ind = blockIdx.z % C;
      int batch_n = blockIdx.z / C;

      if (x_ind >= W || y_ind >= H || c_ind >= C || batch_n >= N) {
            return;
      }

      out[batch_n * H * W * C + c_ind * H * W + y_ind * W + x_ind] =
            (in[batch_n * H * W * C + c_ind * H * W + y_ind * W + x_ind] < 0.0f) ? 0.0f : in[batch_n * H * W * C + c_ind * H * W + y_ind * W + x_ind];

}

// -------------------------------------------------- Helpers --------------------------------------------------------

__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B, int m, int k, int n) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim ((n+TILE_SIZE - 1) / TILE_SIZE, (m+TILE_SIZE -1) / TILE_SIZE);
      float *C;
      int size = m * n;

      cudaMalloc(&C, size * sizeof(float));

      mat_mult_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, m, k, n);

      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(C, size);
}

__host__ std::unique_ptr<DeviceData> conv2d_base(const float *in, const float *kern, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) {

      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H_out + TILE_SIZE - 1) / TILE_SIZE,  (W_out + TILE_SIZE - 1) / TILE_SIZE, C_out * N);

      float *out;
      int size = H_out * W_out * C_out * N;

      cudaMalloc(&out, size * sizeof(float));

      conv2d_kernel<<<gridDim, blockDim>>>(in, kern, out, N, C_in, H, W, H_out, W_out, K, P, S, C_out);

      cudaDeviceSynchronize();

      return std::make_unique<CudaData> (out, size);


}



__host__ std::unique_ptr<DeviceData> pool_base(const float *in, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S, Pool option) {

      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H_out + TILE_SIZE - 1) / TILE_SIZE,  (W_out + TILE_SIZE - 1) / TILE_SIZE, C_in * N);

      float *out;
      int size = H_out * W_out * C_in * N;

      cudaMalloc(&out, size * sizeof(float));

      if(option == Pool::Average) {
            avg_pool2d_kernel<<<gridDim, blockDim>>>(in, out, N, C_in, H, W, H_out, W_out, K, P, S);
      } else if (option == Pool::Max) {
            max_pool2d_kernel<<<gridDim, blockDim>>>(in, out, N, C_in, H, W, H_out, W_out, K, P, S);
      }
      cudaDeviceSynchronize();

      return std::make_unique<CudaData> (out, size);


}

__host__ std::unique_ptr<DeviceData> mat_transpose_base(const float *in, int H, int W, int C, int N) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H + TILE_SIZE - 1) / TILE_SIZE,  (W + TILE_SIZE - 1) / TILE_SIZE, C * N);

      float *out;
      int size = H * W * C * N;

      cudaMalloc(&out, size * sizeof(float));

      mat_transpose_kernel<<<gridDim, blockDim>>>(in, out, H, W, C, N);

      return std::make_unique<CudaData>(out, size);
}

__host__ std::unique_ptr<DeviceData> relu_base(const float *in, int H, int W, int C, int N) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H + TILE_SIZE - 1) / TILE_SIZE,  (W + TILE_SIZE - 1) / TILE_SIZE, C * N);
      float *out;
      int size = H * W * C * N;

      cudaMalloc(&out, size * sizeof(float));

      relu_kernel<<<gridDim, blockDim>>>(in, out, H, W, C, N);

      return std::make_unique<CudaData>(out, size);
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