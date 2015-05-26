//#include "integration.cuh"
#include "helper_math.h"
#include "math_constants.h"

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

	//float3 g = make_float3(0.0f, -0.0003f, 0.0f);
		
	//Velocity_new = (velocity_old + gravity*deltatime)*global_damping
        //vel += g * 0.15;
		vel += params.gravity * dt;
		//vel *= 1.0;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * dt;
		//pos += vel * 0.15;

        // set this to zero to disable collisions with cube sides

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