#ifndef MATRIX_OPS_LIBRARY_CUH
#define MATRIX_OPS_LIBRARY_CUH

const int TILE_SIZE = 16;

__global__ void matMulTiledKernel(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToMatAdd(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToVecAdd(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToScalAdd(const float A[], int A_shape[], const float B, float C[], int C_shape[]);

__global__ void matToMatSub(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToVecSub(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToScalSub(const float A[], int A_shape[], const float B, float C[], int C_shape[]);

__global__ void matToMatMult(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToVecMult(const float A[], int A_shape[], const float B[], int B_shape[], float C[], int C_shape[]);

__global__ void matToScalMult(const float A[], int A_shape[], const float B, float C[], int C_shape[]);

#endif //MATRIX_OPS_LIBRARY_CUH