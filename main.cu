#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define MAIN "_MAIN_:"
#define F1LO "_ODD_:"
#define CTRL "_CRTL_:"

#define ARRAY_SIZE 100000
#define TILE_WIDTH 256

__device__
inline void SWAP(int32_t *_a,int32_t *_b){int32_t __aux; __aux = *_a; *_a = *_b; *_b = __aux;}



/*###################################
###########    KERNELS    ###########
###################################*/

__global__
void odd_even_sort_kernel(int32_t * arr_d, int32_t n){
    int32_t t_position = (blockDim.x * blockIdx.x + threadIdx.x)*2 + 1;// +1 corresponde para evitar el overflow en el 0
    int32_t tid = threadIdx.x*2+1;

        for(int32_t i=0; i<blockDim.x;i++){

        	if ((i&1) && t_position< n-1 && tid < blockDim.x*2-1 ) { // impar
                    if (arr_d[t_position + 1] < arr_d[t_position]) {
                        SWAP(arr_d + t_position, arr_d + t_position + 1);
                    }
                }
                if(!(i&1) && t_position < n && tid < blockDim.x*2){ //par
                    if (arr_d[t_position] < arr_d[t_position-1]) {
                        SWAP(arr_d + t_position, arr_d + t_position - 1);
                    }
                }
                __syncthreads();
        }
}

__global__
void fast_odd_even_sort_kernel(int32_t * arr_d, int32_t n){
    int32_t position = (blockDim.x * blockIdx.x + threadIdx.x)*2 + 1;// +1 corresponde para evitar el overflow en el 0
    int32_t tid = threadIdx.x*2+1;
    __shared__ int32_t sh_arr[2*TILE_WIDTH];
    int32_t bound = blockDim.x*2;
    
    if(position < n){
    	*(sh_arr+tid)=*(arr_d+position);
    	*(sh_arr+tid-1)=*(arr_d+position-1);
    	__syncthreads();
    
    	for(int32_t i=0; i<blockDim.x;i++){

    		if ((i&1) && position< n-1 && tid < bound-1 ) { // impar
    			if (*(sh_arr+tid + 1) < *(sh_arr+tid)) {
    				SWAP(sh_arr + tid, sh_arr + tid + 1);
                }
            }

    		if(!(i&1) && position < n && tid < bound){ //par
                if (*(sh_arr+tid) < *(sh_arr+tid-1)) {
                    SWAP(sh_arr + tid, sh_arr + tid - 1);
                }
            }
            __syncthreads();
    	}

    	*(arr_d+position) = *(sh_arr+tid);
    	*(arr_d+position-1) = *(sh_arr+tid-1);
    }
}


/*##########################################
###########    HOST FUNCTIONS    ###########
##########################################*/


__host__
void odd_even_sort(int32_t * arr, int32_t n){
	int32_t *cuda_d;
	dim3 dimGrid ((uint)((ARRAY_SIZE / TILE_WIDTH)+1), 1, 1);
	dim3 dimBlock (TILE_WIDTH, 1, 1);
	cudaError_t err;
	cudaEvent_t start, stop;
	float mili;

	err = cudaMalloc((void**)&cuda_d, sizeof(int32_t)*ARRAY_SIZE);
	if( err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); // best definition
		exit(EXIT_FAILURE);
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(cuda_d, arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyHostToDevice);

	int32_t j_limit = n*2/TILE_WIDTH;
	int32_t *p_cuda;
	int32_t size;

	printf("%s ordenando..\n",F1LO);
    cudaEventRecord(start);
	for(int32_t j=0;j<j_limit;j++){
		p_cuda = cuda_d + (j&1) * TILE_WIDTH;
		size = n - (j&1) * TILE_WIDTH;
		odd_even_sort_kernel<<<dimGrid, dimBlock>>>(p_cuda, size);
	}
	cudaEventRecord(stop);
	//cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mili, start, stop);
	cudaMemcpy(arr, cuda_d, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

	printf("%s terminanding.. time: %f s\n", F1LO, mili/1000);
	cudaFree(cuda_d);
}


__host__
void fast_odd_even_sort(int32_t * arr, int32_t n){
    int32_t *cuda_d;
    //float tile = TILE_WIDTH , size_t = ARRAY_SIZE;
    //dim3 dimGrid ((uint)ceil(size_t/tile), 1, 1);
    dim3 dimGrid ((uint)((ARRAY_SIZE / TILE_WIDTH)+1), 1, 1);
    dim3 dimBlock (TILE_WIDTH, 1, 1);
    cudaError_t err;
    cudaEvent_t start, stop;
    float mili;

    err = cudaMalloc(&cuda_d, sizeof(int32_t)*ARRAY_SIZE);
    if( err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); // best definition
        exit(EXIT_FAILURE);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(cuda_d, arr, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyHostToDevice);

    int32_t j_limit = n*2/TILE_WIDTH;
    int32_t *p_cuda;
    int32_t size;

    printf("%s ordenando..\n",F1LO);
    cudaEventRecord(start);
    for(int32_t j=0;j<j_limit;j++){
        p_cuda = cuda_d + (j&1) * TILE_WIDTH;
        size = n - (j&1) * TILE_WIDTH;
        fast_odd_even_sort_kernel<<<dimGrid, dimBlock>>>(p_cuda, size);
    }
    cudaEventRecord(stop);
    //cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&mili, start, stop);
    cudaMemcpy(arr, cuda_d, sizeof(int32_t)*ARRAY_SIZE, cudaMemcpyDeviceToHost);

    printf("%s terminanding.. time: %f s\n", F1LO, mili/1000);
    cudaFree(cuda_d);
}


__host__
int control(int32_t *arr, int32_t n){
  for(int32_t i=1; i<n; i++){
    if(arr[i-1] > arr[i]){
        printf("%s I = %d\n", CTRL, i);
        return 1;
    }
  }
  return 0;
}


/*###################################
###########      MAIN     ###########
###################################*/


int main( int argc, char *argv[] ){
    int32_t *arr;
    
    arr = (int32_t*)malloc(sizeof(int32_t)*ARRAY_SIZE);
    
    printf("array size: %d tile: %d\n",ARRAY_SIZE, TILE_WIDTH);

    printf("####   SORT WHIT GLOBAL MEMORY   ####\n" );
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand()%1000;
      //  printf("%d ", arr[i]);
    }
    printf("\n");

    if(control(arr, ARRAY_SIZE)) printf("%s desordenado!! \n",MAIN);
    else printf("%s ok!! \n",MAIN);  
        
    odd_even_sort(arr,ARRAY_SIZE);

    if(control(arr, ARRAY_SIZE)) printf("%s desordenado!! \n",MAIN);
    else printf("%s ok!! \n" ,MAIN);


    printf("####   SORT WHIT SHARED MEMORY   ####\n" );
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand()%1000;
      //  printf("%d ", arr[i]);
    }
    printf("\n");

    if(control(arr, ARRAY_SIZE)) printf("%s desordenado!! \n",MAIN);
    else printf("%s ok!! \n",MAIN);

    fast_odd_even_sort(arr,ARRAY_SIZE);

    if(control(arr, ARRAY_SIZE)) printf("%s desordenado!! \n",MAIN);
    else printf("%s ok!! \n" ,MAIN);

    free(arr);
    printf("\n");

    return 0;
}
