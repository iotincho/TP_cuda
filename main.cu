#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define ARRAY_SIZE 20000
#define TILE_WIDTH 128


__device__
inline void SWAP(int32_t *_a,int32_t *_b){int32_t __aux; __aux = *_a; *_a = *_b; *_b = __aux;}


__global__
void odd_even_sort_kernel(int32_t * arr_d, int32_t n){
    int32_t position = (blockDim.x * blockIdx.x + threadIdx.x)*2 + 1;// +1 corresponde para evitar el overflow en el 0
    int32_t tid = threadIdx.x;
    int32_t j_limit = n*2/blockDim.x;
    int32_t position_limit;
    int32_t t_position;

    for(int32_t j=0;j<j_limit;j++){
        t_position = position + (j&1)*blockDim.x;
        position_limit = n - (j&1)*blockDim.x;

        for(int32_t i=0; i<blockDim.x;i++){
                if ((i&1) && t_position< position_limit-1 && tid < blockDim.x*2-1 ) { // impar
                    if (arr_d[t_position + 1] < arr_d[t_position]) {
                        SWAP(arr_d + t_position, arr_d + t_position + 1);
                    }
                }
                if(!(i&1) && t_position < position_limit && tid < blockDim.x*2){ //par
                    if (arr_d[t_position] < arr_d[t_position-1]) {
                        SWAP(arr_d + t_position, arr_d + t_position - 1);
                    }
                }
                __syncthreads();
        }
    }
}


__host__
int control(int32_t *arr, int32_t n){
  for(int32_t i=1; i<n; i++){
    if(arr[i-1] > arr[i]){
        printf("In :: %d\n",i);
        return 1;
    }
  }
  return 0;
}

int main( int argc, char *argv[] ){
    int32_t arr[ARRAY_SIZE];
    int32_t *cuda_d;

    dim3 dimGrid ((uint)((ARRAY_SIZE / TILE_WIDTH)+1), 1, 1);
    dim3 dimBlock (TILE_WIDTH-1, 1, 1);
    cudaError_t err;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand()%1000;
      //  printf("%d ", arr[i]);
    }
    printf("\n");

    if(control(arr, ARRAY_SIZE)) printf("desordenado!! \n");
    else printf("ok!! \n");

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

    if(control(arr, ARRAY_SIZE)) printf("desordenado!! \n");
    else printf("ok!! \n");
    /*for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", arr[i]);
    }
    */
    printf("\n");

    return 0;
}