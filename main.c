#include <stdlib.h>
#include <cuda.h>

#define IS_EVEN(_value_){return _value_ % 2;}
#define SWAP(_a,_b){int32_t _aux;\
                      _aux = *_a;\
                      *_a = *_b;\
                      *_b = _aux;\
                      return;}


int main(){



}

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
