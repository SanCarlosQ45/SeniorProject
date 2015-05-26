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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

//Enuum used in various switch statements to decide
enum kernel {CUBIC_SPLINE,
	         GAUSSIAN,
             QUINTIC_SPLINE,
             WENDLAND_QUINTIC_SPLINE,
             HARMONIC,
             M6_SPINE,
             W8,
             W10,
             REPULSIVE,
             POLY6};

//Enum used in various switch statements to decide
//which integration method to use, add new one here
enum Type {EULER, VERLET};

//Global simulation parameters, anything not specific to a group
//should be added here
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;

	kernel kernel_type;
	Type integration_type;

	uint timestep;
};

//struct for the sph kernel data to be used by gpu
struct kernel_data{
	int dim;
	double fac;

	double* smoothing;
	double* distances;
	double* function_cache;
	double* gradient_cache;

	double constant_h;
	double distances_dx;

	int has_constant_h;
};


//These are the constants assocaited with a particle group
struct GroupParams{
	float spring;
    float damping;
    float shear;
    float attraction;
};

#endif
