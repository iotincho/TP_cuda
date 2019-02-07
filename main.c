#include <stdlib.h>
#include <cuda.h>

#define ARRAY_SIZE 1024;
#define TILE_WIDTH 64;

#define IS_EVEN(_value_){return _value_ % 2;}
#define SWAP(_a,_b){int32_t _aux;\
                      _aux = *_a;\
                      *_a = *_b;\
                      *_b = _aux;\
                      return;}



__global__
void odd_even_sort_kernel(int32_t * arr, int32_t n){
  int32_t tid = threadIdx.x + blockDim.x * blockId.x +1;// +1 corresponde para evitar el overflow en el 0

  if(tid < n-1){
    for( int i=1 ; i<=n ;i++){
      if(IS_EVEN(i)){
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



int main(void){

int32_t arr[ARRAY_SIZE];
int32_t *cuda_arr;

dim3 dimGrid (ARRAY_SIZE/TILE_WIDTH +1 ,1 ,1);
dim3 dimBlock (TILE_WIDTH,1,1);

for (int i = 0; i < ARRAY_SIZE; i++) {
   arr[i] = rand();
 }

 cudaMalloc(&cuda_arr,sizeof(int32_t)*ARRAY_SIZE);

 cudaMemcpy(cuda_arr, arr, sizeof(int32_t)*ARRAY_SIZE,cudaMempyHostToDevice);

 odd_even_sort_kernel<<<dimGrid,dimBlock>>>(cuda_arr,ARRAY_SIZE);

 cudaDeviceSynchronize();

 cudaMemcpy(arr, cuda_arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

 cudaFree(cuda_arr);

return 0;
}
