/*-----------------------------------------------------------*
 * "sphkernels.h"                                            *
 *-----------------------------------------------------------*
 * The sphkernel object uses polymorphism and inheritence to *
 * create the various sph kernels. The base class provides   *
 * the functions and data fields for use in the various      *
 * kernels.                                                  *
 *-----------------------------------------------------------*
 * Author: Josh Lohse                                        *
 * Date: 11/11/2014                                          *
 *-----------------------------------------------------------*/

#include "sphkernel.cuh"
#include "helper_math.h"
#include "vector_types.h"

#define PI 3.1415

__device__ inline double cuda_h_dim(double h, int dim){
    if (dim == 1)
        return 1/h;
    if (dim == 2)
        return 1/(h*h);

    return 1/(h*h*h);
}

__device__ double cuda_interpolate_function(double rab, kernel_data* kerndata){

	if(rab > 2*kerndata->constant_h)
		return 0.0;
	
	int index_low = (int)floor(rab/kerndata->distances_dx);
	int index_high = index_low + 1;

	double slope = (kerndata->function_cache[index_high] - kerndata->function_cache[index_low])/kerndata->distances_dx;

	return slope * fmod(rab, kerndata->distances_dx) + kerndata->function_cache[index_low];
}

__device__ double cuda_interpolate_gradients(double rab, kernel_data* kerndata){
	if(rab > 2*kerndata->constant_h)
		return 0.0;
	
	int index_low = (int)floor(rab/kerndata->distances_dx);
	int index_high = index_low + 1;

	double slope = (kerndata->gradient_cache[index_high] - kerndata->gradient_cache[index_low])/kerndata->distances_dx;

	return slope * fmod(rab, kerndata->distances_dx) + kerndata->gradient_cache[index_low];
}

__device__ double cuda_gaussian_function(float3 a, float3 b, double fac, kernel_data* kerndata){
	double f = fac * cuda_h_dim(kerndata->constant_h, kerndata->dim);
	double r = sqrt((b.x-a.x)*(b.x-a.x) +(b.y-a.y)*(b.y-a.y)+(b.z-a.z)*(b.z-a.z));
	double q = r/kerndata->constant_h;

	return f*exp(-q*q);
}

__device__ float3 cuda_gaussian_gradient(float3 a, float3 b, double fac, kernel_data* kerndata){
	float3 grad;

	double rab = sqrt((b.x-a.x)*(b.x-a.x) +(b.y-a.y)*(b.y-a.y)+(b.z-a.z)*(b.z-a.z));
	double f = fac*cuda_h_dim(kerndata->constant_h, kerndata->dim);
	float3 r = a - b; //vector subtraction
	double q = rab/kerndata->constant_h;
	double val = 0.0;

	if(q > 1e-14){
		val = -2*q*exp(-q*q)/(rab*kerndata->constant_h);
	}
	grad.x = r.x * (val * f);
	grad.y = r.y * (val * f);
	grad.z = r.z * (val * f);

	return grad;
}

__device__ double gaussian_calc_fac(kernel_data* kerndata){
	if(kerndata->dim == 1)
		return (1.0/((PI*0.5)*kerndata->constant_h));
	if(kerndata->dim == 2)
		return (1.0/(((PI*.5)*kerndata->constant_h)*2));

	return (1.0/(((PI*.5)*kerndata->constant_h)*3));
}