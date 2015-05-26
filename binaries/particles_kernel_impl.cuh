/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "sphkernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;
__constant__ GroupParams* gparams;


/**
	Calculates which cell in the grid the current particle is in, this 
	is used both in the hash table as well as to calculate collisions
	
	@param p a vector containing the current position in space of the particle
	@return the grid position of the particle, so the cell location
*/
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
/**
	Calculates address into hash table of the current particle from its grid
	location rather than space postion
	
	@param gridPos grid position from pervious function
	@return address into hash table to be used when hash function and sort is run
*/
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}
/**
	Function that calls the calcGridPos() and calcGridHash() function for the
	current particle
	
	@param *gridParticleHash pointer to array containing hash value 
	@param *gridParticleIndex pointer to array containing index values for each particle
	@param *pos current position for the particle in vector/coorinate form
	@param numPartiles total number of particles being simulated
	
	@return none
*/
// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
/**
	Function that sorts the particles based on location, so which cell its in, and
	with that finds where the first and last particles are in a cell. This means
	the start and end of a cell in the hash table.
	
	@param *cellStart pointer to an uint that will contain the index to start of a cell
	@param *cellEnd pointer to an uint that will contain the index to the end of a cell
	@param *sortedPos pointer to an array containing sorted position based on particle index
	@param *sortedVel pointer to an array containing sorted velocities based on particle index
	@param *gridParticleHash pointer to hash table that contains the sorted hash tables
	@param *gridParticleIndex pointer to array of sorted particle indexes
	@param *oldPos pointer to array containing positions to be placed in sorted order in sortedPos
	@param *oldVel pointer to array containing velocities to be placed in sorted order in sortedVel
	@param numPartilces constant value containing the number of particles being simulated
*/
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: sorted position array
                                  float4 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }


}

// collide two spheres using DEM method
/**
	Function that calculates the force to be added to particle A's veloctiy later.
	This calculation takes into account the spring, damping, shear, and attraction
	forces specifically their constants.
	@param posA position of particle A in x,y,z
	@param posB position of particle B in x,y,z
	@param velA velocity vector of particle A
	@param velB velocity vecotor of particle B
	@param radiusA radius of particle A, used to calculate collide distance
	@param radiusB radius of particle B, used to calculate collide distance
	@param attraction constant used to calculate the attraction force between particle A and B
	
	@return none
*/
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction,
					  uint group_num,
					  GroupParams* d_gparams_array)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

		GroupParams gp = d_gparams_array[group_num];

		
		 // spring force
        force = -gp.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += gp.damping*relVel;
        // tangential shear force
        force += gp.shear*tanVel;
        // attraction
        force += gp.attraction*relPos;	
		
    }

    return force;
}



// collide a particle against all other particles in a given cell
/**
	Function that iterates over all of the particles in a cell and calculates all
	collisions with the original particle. This function is called for each of the eight
	cells around the center cell where the particle is located. Forces are summed and
	returned to the collideD function.
	
	@param gridPos cell number in grid in x,y,z coordinates
	@param index index of particle being looked at by thread
	@param pos vector containing the position of the current particle in x,y,z
	@param vel vector containing velocity vector of current particle
	@param *oldPos pointer to array containing all of the position vectors indexed by the particle number
	@param *oldVel pointer to array containing all of the velocity vectors indexed by the particle number
	@param *cellStart pointer to array containing the start pointer of each cell in the arrays
	@param *cellEnd pointer to array containing the end pointer of each cell in the arrays
	
	@return force sum total of all forces on particle from other particles in current cell
*/
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  pos, //Need to change to float 4 or pass the group number
                   float3  vel,
                   float4 *oldPos,
                   float4 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd,
				   uint   group_num,
				   GroupParams* d_gparams_array)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(FETCH(oldPos, j)); //Getting position and velocity of other particle
                float3 vel2 = make_float3(FETCH(oldVel, j));

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction, group_num, d_gparams_array);
            }
        }
    }

    return force;
}

/**
	Top function in collision calculation that iterates over all eight cells around 
	the current cell. All forces are calculated for the current particle and are
	eventually added to the new velocity vector.
	
	@param *newVel pointer to array containing the calculated velocity vectors for each particle
	@param *oldPos pointer to array containing the old position coordinates for each particle
	@param *oldVel pointer to array containing the old velocity vectors for each particle
	@param *gridParticleIndex pointer to array containing the sorted list of particle indices
	@param *cellStart pointer to array containing the indecies to where each cell starts
	@param *cellEnd pointer to array containing the indecies to where each cell ends
	@param numParticles the total number of particles being simulated
	
	@return none
*/

__global__
void collideD(float4 *newVel,               // output: new velocity
			  float4 *newForce,				// output: force
              float4 *oldPos,               // input: sorted positions
              float4 *oldVel,               // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
			  GroupParams* d_gparams_array)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float4 pos = FETCH(oldPos, index);
	uint group_num = pos.w;
    float3 vel = make_float3(FETCH(oldVel, index));


    // get address in grid
    int3 gridPos = calcGridPos(make_float3(pos));

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

	//Iterates over all of the 26 cells around the particles cell in the center
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, make_float3(pos), vel, oldPos, oldVel, cellStart, cellEnd, group_num, d_gparams_array); //Sum the forces for each cell
            }
        }
    }

    // collide with cursor sphere
    force += collideSpheres(make_float3(pos), params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f, group_num, d_gparams_array);

    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
	newForce[originalIndex] = make_float4(force, 0.0f);
    newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

#endif
