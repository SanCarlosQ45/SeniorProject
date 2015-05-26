/**
* sphkernel.cuh
*
* created by Josh Lohse
* 11/27/2014
*
* Set of sph kernel function declarations. All additional CUDA gpu sph kernel
* functions should go in sphkernel.cu and where the cpu versions of these go
* in the cpukernel files.
*
* Follow the format of the existing functions for adding new functions, if the
* function needs to be called by the host then use __global__ 
*
*/

#ifndef SPHKERNEL_H
#define SPHKERNEL_H

#include "particles_kernel.cuh"
#include "helper_math.h"

__device__ double cuda_interpolate_function(double rab, kernel_data* kerndata);
__device__ double cuda_interpolate_gradients(double rab,kernel_data* kerndata);

__device__ double cuda_gaussian_function(float3 a, float3 b, double fac, kernel_data* kerndata);
__device__ float3 cuda_gaussian_gradient(float3 a, float3 b, double fac, kernel_data* kerndata);
__device__ double gaussian_calc_fac(kernel_data* kerndata);

//Add new kernel functions here...

#endif