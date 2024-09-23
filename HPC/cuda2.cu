#include <iostream>
#include <chrono>
using namespace std;
__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << "Enter element " << i + 1 << " of the vector: ";
        cin >> vector[i];
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

void sequentialAddition(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N;
    cout << "Enter the size of the vectors: ";
    cin >> N;

    int* A, * B, * C;
    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    initialize(A, vectorSize);
    initialize(B, vectorSize);

    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Sequential addition
    auto start = std::chrono::high_resolution_clock::now();
    sequentialAddition(A, B, C, N);
    auto stop = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "Sequential Addition: ";
    print(C, N);

    // Parallel addition
    start = std::chrono::high_resolution_clock::now();
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
    stop = std::chrono::high_resolution_clock::now();
    auto par_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << "Parallel Addition: ";
    print(C, N);

    cout << "Sequential Addition Time: " << seq_duration.count() << " microseconds" << endl;
    cout << "Parallel Addition Time: " << par_duration.count() << " microseconds" << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}

