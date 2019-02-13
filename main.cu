#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define ARRAY_SIZE 1024
#define TILE_WIDTH 64

//#define IS_EVEN(_value_){return _value_ % 2;}

__device__
inline void SWAP(int32_t *_a,int32_t *_b){int32_t __aux; __aux = *_a; *_a = *_b; *_b = __aux;}


__global__
void odd_even_sort_kernel(int32_t * arr_d, int32_t n){
    int32_t tid = blockDim.x * blockIdx.x + threadIdx.x + 1;// +1 corresponde para evitar el overflow en el 0

    for(int i=0; i<n;i++){
        if(tid < n) {
            if (tid%2) { // porque usa tid? deberia usar i
                if (arr_d[tid] < arr_d[tid-1]) {
                    SWAP(arr_d + tid, arr_d + tid - 1);
                }
            }
            __syncthreads();
            if (!(tid%2)) {
                if (arr_d[tid] < arr_d[tid-1]) {
                    SWAP(arr_d + tid, arr_d + tid - 1);
                }
            }
            __syncthreads();
        }
    }
}

/*  PROPUESTA

    int odd_even;
    for(int i=0; i<n;i++){
    	odd_even=(i%2)*2-1; // odd_even=1 si es impar, =-1 si es par,la idea es ahorrar un if
        if(tid < n) {// puedo sacar este if fuera del for? seria mas efeiciente ejecutarlo una sola vez.
            if (arr_d[tid] < arr_d[tid-1]) {
               SWAP(arr_d + tid, arr_d + tid + odd_even);
            }

            __syncthreads(); // de esta otra forma es necesario?
        }
    }

*/

int control(int32_t *arr, int32_t *n){
  for(int i=1;i<n;i++){
    if(arr[i-1]>arr[i]) return 1;
  }
  return 0;
}

int main( int argc, char *argv[] ){
    int32_t arr[ARRAY_SIZE];
    int32_t *cuda_d;

    dim3 dimGrid ((uint)ceil(ARRAY_SIZE / TILE_WIDTH), 1, 1);
    dim3 dimBlock (TILE_WIDTH, 1, 1);
    cudaError_t err;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand()%1000;
      //  printf("%d ", arr[i]);
    }
    printf("\n");

    err = cudaMalloc(&cuda_d, sizeof(int32_t)*ARRAY_SIZE);
    if( err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); // best definition
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(cuda_d, arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyHostToDevice);

    odd_even_sort_kernel<<<dimGrid, dimBlock>>>(cuda_d, ARRAY_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(arr, cuda_d, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(cuda_d);

    if(control(arr)) printf("desordenado!! \n");
    /*for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", arr[i]);
    }
    */
    printf("\n");

    return 0;
}
