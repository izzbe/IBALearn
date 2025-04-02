#include "CudaData.cuh"
#include <iostream>
#include <cfloat>

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

// -------------------------------------------------------- FORWARD --------------------------------------------------------------
//
//
//
//
//
//
// ---------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------ Decls -------------------------------------------------------------
__host__ std::unique_ptr<DeviceData> elem_wise(const CudaData *A, const CudaData *B, Operation o, int B_grouping, int B_stride);
__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B,  int H, int shared_axis, int W, int N);
__host__ std::unique_ptr<DeviceData> conv2d_base(const float *in, const float *kern, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S, int C_out);
__host__ DeviceData::max_pool_return max_pool_base(const float *in, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S);
__host__ std::unique_ptr<DeviceData> mat_transpose_base(const float *in, int H, int W, int C, int N);
__host__ std::unique_ptr<DeviceData> relu_base(const float *in, int H, int W, int C, int N);

__global__ void avg_pool2d_kernel(const float *in, float *out, int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S);
// ------------------------------------------------- Matrix Ops -------------------------------------------------------
std::unique_ptr<DeviceData> CudaData::elemAdd(const DeviceData *other, int B_grouping, int B_stride) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Add, B_grouping, B_stride);
}

std::unique_ptr<DeviceData> CudaData::elemSub(const DeviceData *other, int B_grouping, int B_stride) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Sub, B_grouping, B_stride);
}

std::unique_ptr<DeviceData> CudaData::elemMult(const DeviceData *other, int B_grouping, int B_stride) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return elem_wise(this, type_check, Operation::Mult, B_grouping, B_stride);
}

std::unique_ptr<DeviceData> CudaData::mat_mult(const DeviceData *other, int H, int shared_axis, int W, int N) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(other);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }

      return mat_mult_base(this, type_check, H, shared_axis, W, N);
}


std::unique_ptr<DeviceData> CudaData::conv2d(const DeviceData *kern, int N, int C_in,
                                             int H, int W, int H_out, int W_out, int K, int P, int S, int C_out) const {
      const CudaData *type_check = dynamic_cast<const CudaData*>(kern);
      if (!type_check) {
            throw std::logic_error("Cannot operate on GPU and CPU allocated memory");
      }
      return conv2d_base(head, type_check->getData(), N, C_in, H, W, H_out, W_out, K, P, S, C_out);
}


std::unique_ptr<DeviceData> CudaData::avg_pool(int N, int C_in, int H, int W, int H_out, int W_out,
                                               int K, int P, int S) const {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H_out + TILE_SIZE - 1) / TILE_SIZE,  (W_out + TILE_SIZE - 1) / TILE_SIZE, C_in * N);

      float *out;
      int size = H_out * W_out * C_in * N;

      cudaMalloc(&out, size * sizeof(float));

      avg_pool2d_kernel<<<gridDim, blockDim>>>(head, out, N, C_in, H, W, H_out, W_out, K, P, S);

      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(out, size);
}

DeviceData::max_pool_return CudaData::max_pool(int N, int C_in, int H, int W, int H_out, int W_out,
                                               int K, int P, int S) const {
      return max_pool_base(head, N, C_in, H, W, H_out, W_out, K, P, S);
}


std::unique_ptr<DeviceData> CudaData::mat_transpose(int H, int W, int C, int N) const {
      return mat_transpose_base(head, H, W, C, N);
}

std::unique_ptr<DeviceData> CudaData::relu(int H, int W, int C, int N) const {
      return relu_base(head, H, W, C, N);
}
// ------------------------------------------------ Kernels ------------------------------------------------------

__global__ void set_index_kernel(float A[], int i, float val) {
	A[i] = val;
}

__global__ void mat_mult_kernel(float A[], float B[], float C[], int H, int shared_axis, int W, int N) {
      int global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
      int global_col = blockIdx.x * TILE_SIZE + threadIdx.x;
      int batch_n = blockIdx.z;

      __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
      __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

      float c_val = 0.0f;

      for (int i = 0; i < (shared_axis + TILE_SIZE - 1) / TILE_SIZE; ++i) {
            if (global_row < H && (i * TILE_SIZE + threadIdx.x) < shared_axis) {
                  shared_A[threadIdx.y][threadIdx.x] = A[batch_n * H * shared_axis + global_row * shared_axis + (i * TILE_SIZE + threadIdx.x)];
            } else {
                  shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (global_col < W && (i * TILE_SIZE + threadIdx.y) < shared_axis) {
                  shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * W + global_col];
            } else {
                  shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (int j = 0; j < TILE_SIZE; ++j) {
                  c_val += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
            }

            __syncthreads();
      }

      if (global_row < H && global_col < W) {
            C[batch_n * H * W + global_row * W + global_col] = c_val;
      }
}

__global__ void mat_add_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size, int B_grouping, int B_stride) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] + B[(global_ind / B_stride) % B_grouping];
      }
}

__global__ void mat_sub_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size, int B_grouping, int B_stride) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] - B[(global_ind / B_stride) % B_grouping];
      }
}

__global__ void mat_element_mult_kernel(float A[], float B[], float C[], int a_size, int b_size, int c_size, int B_grouping, int B_stride) {
      int global_ind = blockIdx.x * blockDim.x + threadIdx.x;

      if (global_ind < c_size && global_ind < a_size) {
            C[global_ind] = A[global_ind] * B[(global_ind / B_stride) % B_grouping];
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


__global__ void max_pool2d_kernel(const float *in, float *out, int *max_inds, int N, int C_in, int H, int W, int H_out, int W_out, int K, int P, int S) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_in;
      int batch_n = blockIdx.z / C_in;

      if (out_x >= W_out || out_y >= H_out || out_c >= C_in || batch_n >= N) return;

      int input_x = out_x * S - P;
      int input_y = out_y * S - P;

      float cur_max = -FLT_MAX;
      float cur_max_ind = 0;

      for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                  int read_x = input_x + j;
                  int read_y = input_y + i;
                  if (read_x >= 0 && read_x < W && read_y >= 0 && read_y < H) {
                        if (in[batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x] > cur_max) {
                              cur_max = in[batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x];
                              cur_max_ind = batch_n * C_in * H * W + out_c * H * W + read_y * W + read_x;
                        }
                  }
            }
      }

      out[batch_n * C_in * H_out * W_out + out_c * H_out * W_out + out_y * W_out + out_x] = cur_max;
      max_inds[batch_n * C_in * H_out * W_out + out_c * H_out * W_out + out_y * W_out + out_x] = cur_max_ind;
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

__host__ std::unique_ptr<DeviceData> mat_mult_base(const CudaData *A, const CudaData *B,  int H, int shared_axis, int W, int N) {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim ((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE -1) / TILE_SIZE, N);
      float *C;
      int size = H * W * N;

      cudaMalloc(&C, size * sizeof(float));

      mat_mult_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, H, shared_axis, W, N);

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



__host__ DeviceData::max_pool_return max_pool_base(const float *in, int N, int C_in,
                                                 int H, int W, int H_out, int W_out, int K, int P, int S) {

      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((H_out + TILE_SIZE - 1) / TILE_SIZE,  (W_out + TILE_SIZE - 1) / TILE_SIZE, C_in * N);

      float *out;
      int *max_inds;
      int size = H_out * W_out * C_in * N;

      cudaMalloc(&out, size * sizeof(float));
      cudaMalloc(&max_inds, size *sizeof(int));
      max_pool2d_kernel<<<gridDim, blockDim>>>(in, out, max_inds, N, C_in, H, W, H_out, W_out, K, P, S);

      cudaDeviceSynchronize();

      std::unique_ptr<int> max_ind_return(max_inds);

      return {std::make_unique<CudaData> (out, size), std::move(max_ind_return)};
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

__host__ std::unique_ptr<DeviceData> elem_wise(const CudaData *A, const CudaData *B, Operation o, int B_grouping, int B_stride) {
      dim3 blockDim(TILE_SIZE * TILE_SIZE);
      dim3 gridDim((A->getSize() + (TILE_SIZE*TILE_SIZE) - 1) / (TILE_SIZE * TILE_SIZE));
      float *C;
      int size = A->getSize() * sizeof(float);

      cudaMalloc(&C, size);

      if (o == Operation::Add) {
            mat_add_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize(), B_grouping, B_stride);
      } else if (o == Operation::Sub) {
            mat_sub_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize(), B_grouping, B_stride);
      } else if (o == Operation::Mult) {
            mat_element_mult_kernel<<<gridDim, blockDim>>>(A->getData(), B->getData(), C, A->getSize(), B->getSize(), A->getSize(), B_grouping, B_stride);
      }

      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(C, A->getSize());

}

// -------------------------------------------------------- BACKWARD --------------------------------------------------------------
//
//
//
//
//
//
// ---------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------decls-------------------------------------------------------------
__global__ void conv2d_backward_kernel(const float *sigma, const float *input,
                                       int C_k, int K, int C_in, int H_in, int W_in,
                                       int C_sigma, int H_sigma, int W_sigma, int P, int S, float *out);

__global__ void rotate_180_kernel(int H, int W, int C, int N, float *in, float *out);

__global__ void conv2d_backward_kernel_wr_input(const float *sigma, const float *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S, float *out);
__global__ void max_pool_backward_kernel_wr_input(const int *max_inds, const float *sigma, float *out,
                                                  int N_in, int C_in, int H_in, int W_in,
                                                  int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                                  int K, int P, int S);

__global__ void avg_pool_backward_wr_input_kernel(const float *sigma,
                                              int N_in, int C_in, int H_in, int W_in,
                                              int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                              int K, int P, int S, float *out);
// ------------------------------------------------- Matrix Ops -------------------------------------------------------
std::unique_ptr<DeviceData> CudaData::conv2d_backward_wr_kernel(const DeviceData *sigma,
                                            int C_k, int K, int H_in, int W_in, int C_in, int C_sigma, int H_sigma, int W_sigma, int P, int S, int N) const {

    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    dim3 gridDim(((K + TILE_SIZE - 1) / TILE_SIZE) * ((K + TILE_SIZE - 1) / TILE_SIZE),
                C_in * C_k,
                N);

    int size = C_k * C_in * K * K;

    float *C;
    cudaMalloc(&C, size * sizeof(float));

    conv2d_backward_kernel<<<gridDim, blockDim>>>(sigma->getData(), head, C_k, K, C_in, H_in, W_in, C_sigma, H_sigma, W_sigma, P, S, C);

    cudaDeviceSynchronize();
    return std::make_unique<CudaData>(C, size);
}

std::unique_ptr<DeviceData> CudaData::conv2d_backward_wr_input(const DeviceData *sigma, const DeviceData *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S) const {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W_in + TILE_SIZE - 1) / TILE_SIZE, (H_in + TILE_SIZE - 1) / TILE_SIZE, C_in_k * sigma_N);

    int size = sigma_N * C_in_k * H_in * W_in;

    float *out;

    cudaMalloc(&out, size * sizeof(float));

    conv2d_backward_kernel_wr_input<<<gridDim, blockDim>>>(sigma->getData(), kernel->getData(), H_in, W_in, K, C_in_k, C_out_k, sigma_H, sigma_W, sigma_C, sigma_N, P, S, out);

    return std::make_unique<CudaData>(out, size);
}

std::unique_ptr<DeviceData> CudaData::max_pool_backward_wr_input(const int *max_inds, const DeviceData *sigma,
                                                            int N_in, int C_in, int H_in, int W_in,
                                                            int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                                            int K, int P, int S) const {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((W_sigma + TILE_SIZE - 1) / TILE_SIZE, (H_sigma + TILE_SIZE - 1) / TILE_SIZE, C_sigma * N_sigma);
      int size = N_in *C_in *H_in * W_in;
      float *out;
      cudaMalloc(&out, sizeof(float) * size);
      cudaMemset(out, 0.0f, sizeof(float) * size);

      max_pool_backward_kernel_wr_input<<<gridDim, blockDim>>>(max_inds, sigma->getData(), out, N_in, C_in, H_in, W_in, N_sigma, C_sigma, H_sigma, W_sigma, K, P, S);
      cudaDeviceSynchronize();

      return std::make_unique<CudaData>(out, size);
}

std::unique_ptr<DeviceData> CudaData::avg_pool_backward_wr_input(const DeviceData *sigma,
                                                             int N_in, int C_in, int H_in, int W_in,
                                                             int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                                             int K, int P, int S) const {
      dim3 blockDim(TILE_SIZE, TILE_SIZE);
      dim3 gridDim((W_in + TILE_SIZE - 1) / TILE_SIZE, (H_in + TILE_SIZE - 1) / TILE_SIZE, C_in * N_in);

      int size = N_in * C_in * H_in * W_in;

      float *out;

      cudaMalloc(&out, size * sizeof(float));

      avg_pool_backward_wr_input_kernel<<<gridDim, blockDim>>>(sigma->getData(), N_in, C_in, H_in, W_in,
                                                                  N_sigma, C_sigma, H_sigma, W_sigma,
                                                                  K, P, S, out);

      return std::make_unique<CudaData>(out, size);
}
// ------------------------------------------------- Kernels ------------------------------------------------------------

__global__ void conv2d_backward_kernel_wr_input(const float *sigma, const float *kernel, int H_in, int W_in, int K, int C_in_k, int C_out_k,
                                                                 int sigma_H, int sigma_W, int sigma_C, int sigma_N, int P, int S, float *out) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % C_in_k;
    int out_n = blockIdx.z / C_in_k;

    if (out_x < 0 || out_y < 0 || out_c < 0 || out_n < 0 || out_x >= W_in || out_y >= H_in || out_c >= C_in_k || out_n >= sigma_N) {
        return;
    }

    float grad_val = 0.0f;
    for (int c = 0; c < C_out_k; c++) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                int current_top_left_x = out_x - i;
                int current_top_left_y = out_y - j;
                if ( (current_top_left_x + P) % S != 0 || (current_top_left_y + P) % S != 0 ) {
                    continue;
                }

                if ( current_top_left_x < -P || current_top_left_y < -P || current_top_left_x > (W_in + 2 * P - K) || current_top_left_y > (H_in + 2 * P - K) ) {
                    continue;
                }

                int sigma_val_x = (current_top_left_x + P) / S;
                int sigma_val_y = (current_top_left_y + P) / S;

                grad_val += kernel[c * C_in_k * K * K + out_c * K * K + j * K + i] *
                    sigma[out_n * sigma_C * sigma_W * sigma_H + c * sigma_W * sigma_H + sigma_val_y * sigma_W + sigma_val_x];

            }
        }
    }
    out[out_n * C_in_k * H_in * W_in + out_c * H_in * W_in + out_y * W_in + out_x] = grad_val;


}

__global__ void conv2d_backward_kernel(const float *sigma, const float *input,
                                       int C_k, int K, int C_in, int H_in, int W_in,
                                       int C_sigma, int H_sigma, int W_sigma, int P, int S, float *out) {
    int block_x = blockIdx.x % ((K + TILE_SIZE - 1) / TILE_SIZE);
    int block_y = blockIdx.x / ((K + TILE_SIZE - 1) / TILE_SIZE);

    int out_x = block_x * blockDim.x + threadIdx.x;
    int out_y = block_y * blockDim.y + threadIdx.y;

    int out_c_in = blockIdx.y % C_in;
    int out_c_k = blockIdx.y / C_in;

    int out_batch_n = blockIdx.z;

    if (out_x >= K || out_y >= K) {
        return;
    }

    float grad_val = 0.0f;
    for (int i = 0; i < W_sigma; i++) {
        for (int j = 0; j < H_sigma; j++) {
            int input_x = i * S - P + out_x;
            int input_y = j * S - P + out_y;

            if (input_x >= W_in || input_y >= H_in || input_x < 0|| input_y < 0) {
                continue;
            }

            grad_val += input[out_batch_n * C_in * H_in * W_in + out_c_in * H_in * W_in + input_y * W_in + input_x] *
                        sigma[out_batch_n * C_sigma * H_sigma * W_sigma + out_c_k * H_sigma * W_sigma + j * W_sigma + i];
        }
    }
    atomicAdd(&out[ out_c_k * C_in * K * K + out_c_in * K * K + out_y * K + out_x], grad_val);

}

__global__ void max_pool_backward_kernel_wr_input(const int *max_inds, const float *sigma, float *out,
                                                  int N_in, int C_in, int H_in, int W_in,
                                                  int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                                  int K, int P, int S) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_sigma;
      int out_n = blockIdx.z / C_sigma;

      if (out_x < 0 || out_y < 0 || out_c < 0 || out_n < 0 || out_x >= W_sigma || out_y >= H_sigma || out_c >= C_sigma || out_n >= N_sigma) {
            return;
      }

      int ind_pos = max_inds[out_n * C_sigma * H_sigma * W_sigma + out_c * H_sigma * W_sigma + out_y *W_sigma + out_x];
      float sigma_val = sigma[out_n * C_sigma * H_sigma * W_sigma + out_c * H_sigma * W_sigma + out_y *W_sigma + out_x];

      atomicAdd(&out[ind_pos], sigma_val);

}

__global__ void avg_pool_backward_wr_input_kernel(const float *sigma,
                                              int N_in, int C_in, int H_in, int W_in,
                                              int N_sigma, int C_sigma, int H_sigma, int W_sigma,
                                              int K, int P, int S, float *out) {
      int out_x = blockIdx.x * blockDim.x + threadIdx.x;
      int out_y = blockIdx.y * blockDim.y + threadIdx.y;
      int out_c = blockIdx.z % C_in;
      int out_n = blockIdx.z / C_in;

      if (out_x < 0 || out_y < 0 || out_c < 0 || out_n < 0 || out_x >= W_in || out_y >= H_in || out_c >= C_in || out_n >= N_in) {
            return;
      }

      float K_square = K * K;

      float grad_val = 0.0f;
      for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                  int current_top_left_x = out_x - i;
                  int current_top_left_y = out_y - j;
                  if ( (current_top_left_x + P) % S != 0 || (current_top_left_y + P) % S != 0 ) {
                        continue;
                  }

                  if ( current_top_left_x < -P || current_top_left_y < -P || current_top_left_x > (W_in + 2 * P - K) || current_top_left_y > (H_in + 2 * P - K) ) {
                        continue;
                  }

                  int sigma_val_x = (current_top_left_x + P) / S;
                  int sigma_val_y = (current_top_left_y + P) / S;

                  grad_val += (1 / K_square) * sigma[ out_n * N_sigma * C_sigma * H_sigma * W_sigma +
                                                      out_c * H_sigma * W_sigma +
                                                      sigma_val_y * W_sigma +
                                                      sigma_val_x];

            }
      }
      out[out_n * C_in * H_in * W_in + out_c * H_in * W_in + out_y * W_in + out_x] = grad_val;


}