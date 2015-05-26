#include "cpukernel.h"

using namespace std;

inline double h_dim(double h, int dim){
    if (dim == 1)
        return 1/h;
    if (dim == 2)
        return 1/(h*h);

    return 1/(h*h*h);
}

double gaussian_function(float3 a, float3 b, double h, double fac, int dim){
	double f = fac * h_dim(h, dim);
	double r = sqrt((b.x-a.x)*(b.x-a.x) +(b.y-a.y)*(b.y-a.y)+(b.z-a.z)*(b.z-a.z));
	double q = r/h;

	return f*exp(-q*q);
}

//Add other kernel functions here

double __gradient(float3 a, float3 b, double h, double fac, int dim){
	double rab = sqrt((b.x-a.x)*(b.x-a.x) +(b.y-a.y)*(b.y-a.y)+(b.z-a.z)*(b.z-a.z));

	double f = fac*h_dim(h, dim);
	double q = rab/h;
	double val = 0.0;

	if(q >= 1.0 && q < 2.0){
		val = -0.75 * (2-q) * (2-q)/(h*rab);
		
	}
	else if( q > 1e-14){
		val = 3.0*(0.75*q-1)/(h*h);
	}
	
	return val * fac;
}