#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define ARRAY_SIZE 1024
#define TILE_WIDTH 64

//#define IS_EVEN(_value_){return _value_ % 2;}

__device__
inline void SWAP(int32_t *_a,int32_t *_b){int32_t _aux; _aux = *_a; *_a = *_b; *_b = _aux;}


__global__
void odd_even_sort_kernel(int32_t * arr, int32_t n){
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x +1;// +1 corresponde para evitar el overflow en el 0

    if(tid < n-1){
        for( int i=1 ; i<=n ;i++){
        if(i%2){
            if(arr[tid] < arr[tid-1])
                SWAP(arr,arr-1);
            }
            else{
                if(arr[tid] < arr[tid+1])
                    SWAP(arr,arr+1);
            }
        }
    }
}



int main( int argc, char *argv[] ){
    int32_t arr[ARRAY_SIZE];
    int32_t *cuda_arr;

    dim3 dimGrid (ARRAY_SIZE/TILE_WIDTH +1, 1,1);
    dim3 dimBlock (TILE_WIDTH, 1, 1);

    for (int i = 0; i < ARRAY_SIZE; i++) {
    arr[i] = rand()%1000;
    }

    cudaMalloc(&cuda_arr, sizeof(int32_t)*ARRAY_SIZE);

    cudaMemcpy(cuda_arr, arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyHostToDevice);

    odd_even_sort_kernel<<<dimGrid, dimBlock>>>(cuda_arr, ARRAY_SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(arr, cuda_arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(cuda_arr);

    //printf("termine , primero = %i, %i, ultimo = %i",arr[0],arr[1],arr[ARRAY_SIZE -1]);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%i\n", arr[i]);
    }
    return 0;
}
