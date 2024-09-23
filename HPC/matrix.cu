#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void multiply(int* A, int* B, int* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void initialize(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        std::cout << "Enter element " << i + 1 << ": ";
        std::cin >> matrix[i];
    }
}

void print(int* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::cout << matrix[row * cols + col] << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void sequentialMultiply(int* A, int* B, int* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

int main() {
    int M, N, K;
    std::cout << "Enter the number of rows and columns of the first matrix: ";
    std::cin >> M >> N;
    std::cout << "Enter the number of columns of the second matrix: ";
    std::cin >> K;

    int* A, * B, * C;
    int matrixSize = M * K;
    size_t matrixBytes = matrixSize * sizeof(int);
    A = new int[M * N];
    B = new int[N * K];
    C = new int[M * K];

    initialize(A, M, N);
    initialize(B, N, K);

    std::cout << "Matrix A: \n";
    print(A, M, N);
    std::cout << "Matrix B: \n";
    print(B, N, K);

    int* X, * Y, * Z;
    cudaMalloc(&X, M * N * sizeof(int));
    cudaMalloc(&Y, N * K * sizeof(int));
    cudaMalloc(&Z, M * K * sizeof(int));

    cudaMemcpy(X, A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, N * K * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y); // Calculate grid dimensions

    // Sequential multiplication
    auto start = std::chrono::high_resolution_clock::now();
    sequentialMultiply(A, B, C, M, N, K);
    auto stop = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Sequential Multiplication of matrix A and B: \n";
    print(C, M, K);

    // Parallel multiplication
    start = std::chrono::high_resolution_clock::now();
    multiply<<<blocks, threads>>>(X, Y, Z, M, N, K);
    cudaDeviceSynchronize(); // Synchronize device
    cudaMemcpy(C, Z, M * K * sizeof(int), cudaMemcpyDeviceToHost);
    stop = std::chrono::high_resolution_clock::now();
    auto par_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Parallel Multiplication of matrix A and B: \n";
    print(C, M, K);

    std::cout << "Sequential Multiplication Time: " << seq_duration.count() << " microseconds" << std::endl;
    std::cout << "Parallel Multiplication Time: " << par_duration.count() << " microseconds" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}

