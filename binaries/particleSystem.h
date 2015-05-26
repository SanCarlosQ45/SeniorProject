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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <cuda_runtime.h>
#include <cuda.h>

#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "sphkernel.cuh"
#include "cpukernel.h"
#include <string>
#include <vector>

using namespace std;

struct group{
	string name;
	string color;
	string type;
	int rad;
	int width;
	int length;
	int height;
	float coord[3];
	float vel[3];
	int particle_count;
};

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, int* pgn, vector<group> gv, GroupParams* gparams_array, SimParams* m_);
        ~ParticleSystem();	

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
			CONFIG_TEST,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
			FORCE
        };

		double run_function(float3 pa, float3 pb);
		double run_gradient(float3 pa, float3 pb);
		void init_cache(const int n);
		void init_sph_kernel(int dim, double constant_h);

		void set_pgn(int* pgn){ particle_group_number = pgn; }
        void update(float deltaTime, GroupParams* d_gparams_array);
        void reset(ParticleConfig config);
		

        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const float *data, int start, int count);

        int    getNumParticles() const
        {
            return m_numParticles;
        }

        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }

        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }
        void *getCudaColorVBO()            const
        {
            return (void *)m_cudaColorVBO;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count, float stepSize, uint &stepNum);

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }

        void setColliderPos(float3 x)
        {
            m_params.colliderPos = x;
        }

		void set_integration_type(Type t){
			m_params.integration_type = t;
		}

		void set_kernel_type(kernel k){
			m_params.kernel_type = k;
		}

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        float3 getColliderPos()
        {
            return m_params.colliderPos;
        }
        float getColliderRadius()
        {
            return m_params.colliderRadius;
        }
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }

        void addSphere(int index, int limit, float *pos, float *vel, int r, float spacing, int groupnum);

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(int start, uint *size, float* pos, float* vel, float spacing, float jitter, int group);

    protected: // data
        bool m_bInitialized, m_bUseOpenGL;
        uint m_numParticles;
		vector<group> groupvec;
		int* particle_group_number;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities
		float *m_hForce;

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;
		float *m_dForce;

        float *m_dSortedPos;
        float *m_dSortedVel;

		kernel_data m_kerndata;
		kernel_data* m_dkerndata;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint   m_colorVBO;          // vertex buffer object for colors

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

        // params
        SimParams m_params;
		GroupParams* m_gparams;
        uint3 m_gridSize;
        uint m_numGridCells;
        StopWatchInterface *m_timer;
        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
