/**
* integration.h
* 
* Position integration functions for GPU code. Currently only
* euler integration is implemented which was what the original
* code used. 
*
* If new function is needed, just copy the euler function since
* the wall collision if statements are needed. Follow the quick
* start guide for the steps to add the functions.
*
* created by Josh Lohse
* 11/27/2014
*
*/

#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

/**
* euler()
*
* Function for doing the position integration using the euler method. This is the same as the existing function
* but allows for different methods to be selected. Make sure to copy the if statements to any new funtion since
* these handle wall collisions. 
*
* @param p array contaning the position vectors of each particle where the index is the particle number
* @param v array containing the velocity vectors for each particle where the index is the particle number
* @param dt timestep for each frame, taken from init file
* @param numParticles the total number of particles in the simulation
*
* created by Josh Lohse
* 11/27/2014
*/
__global__ void euler(float4* p,
					  float4* v,
					  float dt,
					  int numParticles){

	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

	float4 posData = p[index];
    float4 velData = v[index];

	float3 pos = make_float3(posData.x, posData.y, posData.z);
	float3 vel = make_float3(velData.x, velData.y, velData.z);
		
	//Velocity_new = (velocity_old + gravity*deltatime)*global_damping
	vel += params.gravity * dt;
    vel *= params.globalDamping;

    // new position = old position + velocity * deltaTime
    pos += vel * dt;
	//pos += vel * 0.15;

	//Section checks if particle is in contact with a wall, if it is then
	//apply a damping force to the velocity perpendicular to the wall
    if (pos.x > 1.0f - params.particleRadius)
    {
        pos.x = 1.0f - params.particleRadius;
        vel.x *= params.boundaryDamping;
    }

    if (pos.x < -1.0f + params.particleRadius)
    {
        pos.x = -1.0f + params.particleRadius;
        vel.x *= params.boundaryDamping;
    }

    if (pos.y > 1.0f - params.particleRadius)
    {
        pos.y = 1.0f - params.particleRadius;
        vel.y *= params.boundaryDamping;
    }

    if (pos.z > 1.0f - params.particleRadius)
    {
        pos.z = 1.0f - params.particleRadius;
        vel.z *= params.boundaryDamping;
    }

    if (pos.z < -1.0f + params.particleRadius)
    {
        pos.z = -1.0f + params.particleRadius;
        vel.z *= params.boundaryDamping;
    }

    if (pos.y < -1.0f + params.particleRadius)
    {
        pos.y = -1.0f + params.particleRadius;
        vel.y *= params.boundaryDamping;
	}
	
	p[index] = make_float4(pos, posData.w);
	v[index] = make_float4(vel, velData.w);
}

//Add other methods here

#endif