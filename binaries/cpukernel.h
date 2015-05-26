/**
* cpukernel.h
*
* These are the set of CPU functions to initialize the function_cache and gradient_cache arrays.
* These may not be necessary.
*
* created by Josh Lohse
* 11/27/2014
*
*/

#ifndef CPUKERNEL_H
#define CPUKERNEL_H

#include "helper_math.h"

/**
* gaussian_function()
*
* The gaussian cpu kernel function, used to fill function_cache if gaussian method is selected
*
* @param a 3d point for particle a
* @param b 3d point for particle b
* @param h used to initialize constant_h which descides whether theses arrays are filled in the first place
* @param fac 
* @param dim the number of kernel functions to be called
*
* created by Josh Lohse
* 11/27/2014
*/
double gaussian_function(float3 a, float3 b, double h, double fac, int dim);

//Add other kernel functions here


/**
* gaussian_function()
*
* The general gradient function used by all methods, used to fill gradient_cache
*
* @param a 3d point for particle a
* @param b 3d point for particle b
* @param h used to initialize constant_h which descides whether theses arrays are filled in the first place
* @param fac 
* @param dim the number of kernel functions to be called
*
* created by Josh Lohse
* 11/27/2014
*/
double __gradient(float3 a, float3 b, double h, double fac, int dim);

//Add other kernel gradients here

#endif