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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

using namespace std;

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

extern struct group;

/**
* ParticleSystem()
*
* Initializes the particle system object as well as all of the simulation constants. 
* This includes setting the groupParams and simParams structs to the pointers inside
* the class definition.
*
* @param numParticles the total number of particles in the simulation
* @param gridsize the number of cells along one of the dimensions of the cube
* @param bUseOpenGL bool to turn on/off the graphical portion of the simulation
* @param pgn array of ints containing the group number assocated with each particle
*            based on the index
* @param gv vector containing all of the particle groups and their constants, redundent 
*           but makes iteration easier
* @param gparams_array array containing all of the particle groups and their constants, this
*                      is what is passed to the GPU since only primative types are allowed
* @param m_simparams pointer to the SimParam struct containing the global constants, this is what
*                    is passed to the GPU
*
* modified by Josh Lohse
* 10/27/2014
*/
ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, int* pgn, vector<group> gv, GroupParams* gparams_array, SimParams* m_simparams) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
	m_hForce(0),
	m_dForce(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1),
	m_gparams(gparams_array)
{
	particle_group_number = pgn;
	groupvec = gv;
	m_params = *m_simparams;

    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

	setParameters(&m_params);
    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem(){
    _finalize();
    m_numParticles = 0;
}

uint ParticleSystem::createVBO(uint size){
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

/**
* colorRamp()
*
* Takes a string with the color name and assigns the appropriate rgb values to the float 
* array.
*
* @param t unnecessary float, needs to be gotten rid of
* @param r array to have the rgb values written to
* @param color string containing the color from the initfile
*
* modified by Josh Lohse
* 7/12/2014
*/
void colorRamp(float t, float *r, string color){
	if(color == "Red"){
		r[0] = 1.0;
		r[1] = 0.0;
		r[2] = 0.0;
	}
	else if(color == "Green"){
		r[0] = 0.0;
		r[1] = 1.0;
		r[2] = 0.0;
	}
	else if(color == "Blue"){
		r[0] = 0.0;
		r[1] = 0.0;
		r[2] = 1.0;
	}
	else if(color == "Orange"){
		r[0] = 1.0;
		r[1] = 0.5;
		r[2] = 0.0;
	}
	else if(color == "Yellow"){
		r[0] = 1.0;
		r[1] = 1.0;
		r[2] = 0.0;
	}
	else if(color == "Purple"){
		r[0] = 0.65;
		r[1] = 0.0;
		r[2] = 0.65;
	}
	else{
		r[0] = 1.0;
		r[1] = 0.0;
		r[2] = 0.0;
	}
}


/**
* _initialize()
*
* Initializes the particle systems various arrays as well as most of the global constants.
* If a new array is needed in the simulation then it should be allocated in this function
* and its pointer declaration is in the header file of this class.
*
* @param numParticles the total number of particles in the simulation
*
* modified by Josh Lohse
* 7/21/2014
*/
void ParticleSystem::_initialize(int numParticles){
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
	m_hForce = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
	memset(m_hForce, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL){
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    }
    else{
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    }

    allocateArray((void **)&m_dVel, memSize);

	allocateArray((void **)&m_dForce, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL){
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;
		string color;
		group g;
		int groupid =0;
		
        for (uint i=0; i<m_numParticles; i++){
            float t = i / (float) m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;

#else
			g = groupvec.at(particle_group_number[i]);
			if(groupid != particle_group_number[i]){
				groupid = particle_group_number[i];
			}

			color = g.color;
            colorRamp(t, ptr, color);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else{
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

    sdkCreateTimer(&m_timer);
    m_bInitialized = true;
}

/**
* run_function()
*
* A switch statement that takes enum kernel_type and descides which CPU sph kernel
* function to run. This is needed to fill the function_cache array at initialization.
* Whether these will actually be needed during calculation is another issue.
*
* @param pa a 3d point of particle a
* @param pb a 3d point of particle b
*
* Created by Josh Lohse
* 11/24/2014
*/
double ParticleSystem::run_function(float3 pa, float3 pb){
	double result= 0;
	switch(m_params.kernel_type){
	case CUBIC_SPLINE:
		break;
	case GAUSSIAN:
		result = gaussian_function(pa,pb,m_kerndata.constant_h,m_kerndata.fac, m_kerndata.dim);
		break;
	case QUINTIC_SPLINE:
		break;
	case WENDLAND_QUINTIC_SPLINE:
		break;
	case HARMONIC:
		break;
	case M6_SPINE:
		break;
	case W8:
		break;
	case W10:
		break;
	case REPULSIVE:
		break;
	case POLY6:
		break;
	}

	return result;
}

/**
* run_gradient()
*
* A switch statement that takes enum kernel_type and descides which CPU sph kernel
* function to run. This is needed to fill the gradient_cache array at initialization.
* Whether these will actually be needed during calculation is another issue.
*
* @param pa a 3d point of particle a
* @param pb a 3d point of particle b
*
* Created by Josh Lohse
* 11/24/2014
*/
double ParticleSystem::run_gradient(float3 pa, float3 pb){
	double result;
	switch(m_params.kernel_type){
	case CUBIC_SPLINE:
		break;
	case GAUSSIAN:
		result = __gradient(pa,pb,m_kerndata.constant_h,m_kerndata.fac, m_kerndata.dim);
		break;
	case QUINTIC_SPLINE:
		break;
	case WENDLAND_QUINTIC_SPLINE:
		break;
	case HARMONIC:
		break;
	case M6_SPINE:
		break;
	case W8:
		break;
	case W10:
		break;
	case REPULSIVE:
		break;
	case POLY6:
		break;
	}

	return result;
}

/**
* init_cache()
*
* Function that fills the function_cache and gradient cache arrays at the start
* of the simulation. 
*
* @param n an integer value of the size of the arrays to be initialized
*
* Created by Josh Lohse
* 11/24/2014
*/
void ParticleSystem::init_cache(const int n){
	double* distances = new double[n];
	m_kerndata.function_cache = new double[n];
	m_kerndata.gradient_cache = new double[n];

	for(int i=0; i < n; ++i){
		distances[i] = 2*m_kerndata.constant_h;
	}
	//Fill array
	m_kerndata.distances_dx = distances[1] - distances[0];

	float3 pa = make_float3(0); //3dim point container
	float3 pb = make_float3(0);

			
	for(int i = 0; i < n; ++i){
		pb.x = distances[i]; 
		m_kerndata.function_cache[i] = run_function(pa, pb);
		m_kerndata.gradient_cache[i] = run_gradient(pa, pb);
	}
	m_kerndata.has_constant_h = true;			
}

/**
* init_sph_kernel()
*
* Initializes the constants as well as the function_cache and gradient cache arrays.
* It then handles the copying of the struct containing these onto the GPU
*
* @param dim the number of kernel functions so be run at each calculation
* @param constant_h used to descide whether the arrays are filled or not
*
* Created by Josh Lohse
* 11/24/2014
*/
void ParticleSystem::init_sph_kernel(int dim, double constant_h){
			m_kerndata.dim = dim;
			int n = 1000001;
			m_kerndata.distances_dx = -1;

			m_kerndata.has_constant_h = 0;
			m_kerndata.constant_h = constant_h;

			if(constant_h > 0){
				init_cache(n);
			}
			
			//Set of mallocs to copy kernel_data to GPU
			double* fun_cache;
			double* grad_cache;

			cudaMalloc(&m_dkerndata,sizeof(kernel_data)); //First allocate the struct and copy host struct to this location on the GPU
			cudaMemcpy(m_dkerndata,&m_kerndata,sizeof(kernel_data),cudaMemcpyHostToDevice);

			cudaMalloc(&fun_cache, sizeof(double)*n); //Allocate the arrays on the gpu
			cudaMalloc(&grad_cache, sizeof(double)*n);

			cudaMemcpy(fun_cache, m_kerndata.function_cache, sizeof(double)*n, cudaMemcpyHostToDevice); //Copy the host array data onto the gpu at those locations
			cudaMemcpy(grad_cache, m_kerndata.gradient_cache, sizeof(double)*n, cudaMemcpyHostToDevice);

			cudaMemcpy(&(m_dkerndata->function_cache),&fun_cache, sizeof(double*), cudaMemcpyHostToDevice); //Associate these arrays with the originally allocated struct
			cudaMemcpy(&(m_dkerndata->gradient_cache),&grad_cache, sizeof(double*), cudaMemcpyHostToDevice); //by copying the address to the pointer in the struct
		}

/**
* _finalize()
*
* A function that is run when the simulation is finished. Destroys all of the allocated 
* arrays both on the gpu and the host. If a new array is create make sure to delete + free
* it.
*
* modified by Josh Lohse
* 7/24/2014
*/
void ParticleSystem::_finalize(){
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
	delete [] m_hForce;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dVel);
	freeArray(m_dForce);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL){
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else{
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

/**
* update()
*
* The main function that calls all of the functions in particleSystem_cuda.cu which
* in turn call the GPU kernel functions. If a new CUDA kernel function is added then
* create and call a new function from particleSystem_cuda.cu from here.
*
* @param deltaTime time step length taken from init file
* @param d_gparams_array pointer to the group params array that must be passed by kernel
*                        functions if they need a particles group constants
*
* modified by Josh Lohse
* 11/20/2014
*/
void ParticleSystem::update(float deltaTime, GroupParams* d_gparams_array){
    assert(m_bInitialized);
    float *dPos;

    if (m_bUseOpenGL){
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    }
    else{
        dPos = (float *) m_cudaPosVBO;
    }
    // integrate
	integrate(dPos,m_dVel,deltaTime,m_numParticles,m_params.integration_type);
	
    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_dVel,
        m_numParticles,
        m_numGridCells);

    // process collisions

    collide(
        m_dVel,
		m_dForce,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells,
		d_gparams_array);

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL){
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

/**
* dumpParticles()
*
* Any data needed to be written to an external file should be done here. Follow the existing format
* for any new arrays that have been added and need to be dumped.
*
* @param start starting point in the array to be dumped, currently always zero
* @param count the ending point of the array to be dumped, if an array is less than the numParticles
*              then this will have to be modified or another function should be added
* @param stepSize the current deltaTime of the simulation
* @param stepNum the current number of steps since start, this is multiplied by stepSize to get current time
*
* Created by Josh Lohse
* 4/10/2014
*/
void ParticleSystem::dumpParticles(uint start, uint count, float stepSize, uint &stepNum){

	ofstream posFile;
	ofstream velFile;
	ofstream forceFile;

	posFile.open("posdata.txt", fstream::app);
	velFile.open("veldata.txt", fstream::app);
	forceFile.open("forcedata.txt", fstream::app);

    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);
	copyArrayFromDevice(m_hForce, m_dForce, 0, sizeof(float)*4*count);

	for(uint i=start; i<=count-1; i++){ 
		posFile   << stepNum * stepSize << setw(12) << m_hPos[i*4] << setw(12) << m_hPos[i*4+1] << setw(12)
					<< m_hPos[i*4+2] << setw(12) << m_hPos[i*4+3] << setw(12);
		velFile   << stepNum * stepSize << setw(12) << m_hVel[i*4] << setw(12) << m_hVel[i*4+1] << setw(12)
					<< m_hVel[i*4+2] << setw(12);
		forceFile << stepNum * stepSize << setw(12) << m_hForce[i*4] << setw(12) << m_hForce[i*4+1] << setw(12)
					<< m_hForce[i*4+2] << setw(12);
	}
	posFile << endl;
	velFile << endl;
	forceFile << endl;

	++stepNum;
	posFile.close();
	velFile.close();
	forceFile.close();
	//myfile.close();
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            cuda_vbo_resource = m_cuda_posvbo_resource;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;

		case FORCE:
			hdata = m_hForce;
			ddata = m_dForce;
			break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
            }
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;

		case FORCE:
			copyArrayToDevice(m_dForce, data, start*4*sizeof(float), count*4*sizeof(float));
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(int start, uint *size, float* pos, float* vel, float spacing, float jitter, int group)
{
    srand(1973);
	int t=0;
    for (uint z=0; z<size[2]; z++){
        for (uint y=0; y<size[1]; y++){
            for (uint x=0; x<size[0]; x++){

                uint i = (z*size[1]*size[0]) + (y*size[0]) + x + start;
                    m_hPos[i*4] = pos[0] + (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = pos[1] + (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = pos[2] + (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = group;
					
                    m_hVel[i*4] = vel[0];
                    m_hVel[i*4+1] = vel[1];
                    m_hVel[i*4+2] = vel[2];
                    m_hVel[i*4+3] = 0.0f;

					m_hForce[i*4] = 0.0f;
                    m_hForce[i*4+1] = 0.0f;
                    m_hForce[i*4+2] = 0.0f;
                    m_hForce[i*4+3] = 0.0f;
					++t;
            }
        }
    }
}

void
ParticleSystem::addSphere(int start, int limit, float *pos, float *v, int r, float spacing, int groupnum){
    uint index = 0;
    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
				
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < limit))
                {
                    m_hPos[index*4+start]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1+start] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2+start] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3+start] = groupnum;					
					
					m_hVel[index*4+start]   = v[0];
                    m_hVel[index*4+1+start] = v[1];
                    m_hVel[index*4+2+start] = v[2];
                    m_hVel[index*4+3+start] = 0.0f;
					
                    ++index;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
	setArray(FORCE, m_hForce, start, index);
}

/**
* reset()
*
* This is the function that takes the group vector and places each object in the environment.
* This means that is fills the position and velocity vectors with particles corresponding to
* the initial position of the shape
*
* @param config enum containing type of reset to be done
*
* modified by Josh Lohse
* 4/25/2014
*/
void ParticleSystem::reset(ParticleConfig config){
    switch (config){
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    m_hPos[p++] = 2 * (point[0] - 0.5f);
                    m_hPos[p++] = 2 * (point[1] - 0.5f);
                    m_hPos[p++] = 2 * (point[2] - 0.5f);
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
				float c[3] = {0, 0, 0};
                initGrid(0,gridSize, c,c, m_params.particleRadius*2.0f, jitter, 0);
            }
            break;
		case CONFIG_TEST:
			{
				int start = 0;
				group g;
				float* vel = 0;
				float jitter = m_params.particleRadius*0.01f;
				uint gridSize[3];

				for(int i=0; i < groupvec.size(); ++i){
					g = groupvec.at(i);
					
					if(g.type == "Sphere"){
						addSphere(start,g.particle_count+start, g.coord, g.vel, g.rad, m_params.particleRadius*2.0f, i);
					}
					else if(g.type == "Square"){
						gridSize[0] = gridSize[1] = gridSize[2] = g.length;
						initGrid(start, gridSize, g.coord, g.vel, m_params.particleRadius*2.0f, jitter, i);
					}
					else if(g.type == "Rectangle"){
						gridSize[0] = g.length;
						gridSize[1] = g.width;
						gridSize[2] = g.height;
						initGrid(start, gridSize, g.coord, g.vel, m_params.particleRadius*2.0f, jitter, i);
					}
					start += g.particle_count;
				}
			}
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
	setArray(FORCE, m_hForce, 0, m_numParticles);
}
