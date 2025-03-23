#include "CudaData.cuh"
#include <iostream>
#include <cfloat>

// -------------------------------------------------------- BACKWARD --------------------------------------------------------------
//
//
//
//
//
//
// ---------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------- Matrix Ops -------------------------------------------------------





// ------------------------------------------------- Kernels ------------------------------------------------------------

__global__ void conv2d_backward_kernel(const float *sigma, const float *input, const float *kern,
                                       int N, int C_k, int H_k, int W_k, int H_in, int W_in, int C_in, int P, int S, float *out) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y +threadIdx.y;
    int out_c = blockIdx.z % C;
    int batch_n = blockIdx.z / C;


}