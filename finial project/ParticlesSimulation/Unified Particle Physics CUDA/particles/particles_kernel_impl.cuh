/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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
//#include "ParticleMesh.h"
#include <nvMatrix.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
using namespace cub;
using namespace nv;
//#include "Object.h"
//#include "cutil.h"
#include "sm_20_atomic_functions.h"
//#include "sm_11_atomic_functions.h"
//#include "sm_12_atomic_functions.h"

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
#ifndef OBJECTTYPE
#define OBJECTTYPE
typedef enum ObjectType{ Particle, Rigid, Deformation_linear, Deformation_linear_plasticity, Deformation_quad, Deformation_quad_plasticity }ObjectType;
#endif
#ifndef ITERATIONTYPE
#define ITERATIONTYPE
typedef enum IterationType{ Pre_Stablization, Main_Constaint }IterationType;
#endif
struct integrate_functor
{
	float deltaTime;

	__host__ __device__
		integrate_functor(float delta_time) : deltaTime(delta_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
			volatile float4 posData = thrust::get<0>(t);
			volatile float4 velData = thrust::get<1>(t);
			float3 pos = make_float3(posData.x, posData.y, posData.z);
			float3 vel = make_float3(velData.x, velData.y, velData.z);

			vel += params.gravity * deltaTime;
			vel *= params.globalDamping;

			// new position = old position + velocity * deltaTime
			pos += vel * deltaTime;

			// set this to zero to disable collisions with cube sides
#if 1

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

#endif

			if (pos.y < -1.0f + params.particleRadius)
			{
				pos.y = -1.0f + params.particleRadius;
				vel.y *= params.boundaryDamping;
			}

#if 1

			if (pos.x > 2.0f - params.particleRadius)
			{
				pos.x = 2.0f - params.particleRadius;
				vel.x *= params.boundaryDamping;
			}

			if (pos.x < -2.0f + params.particleRadius)
			{
				pos.x = -2.0f + params.particleRadius;
				vel.x *= params.boundaryDamping;
			}

			if (pos.y > 2.0f - params.particleRadius)
			{
				pos.y = 2.0f - params.particleRadius;
				vel.y *= params.boundaryDamping;
			}

			if (pos.z > 2.0f - params.particleRadius)
			{
				pos.z = 2.0f - params.particleRadius;
				vel.z *= params.boundaryDamping;
			}

			if (pos.z < -2.0f + params.particleRadius)
			{
				pos.z = -2.0f + params.particleRadius;
				vel.z *= params.boundaryDamping;
			}

#endif

			if (pos.y < -1.0f + params.particleRadius)
			{
				pos.y = -1.0f + params.particleRadius;
				vel.y *= params.boundaryDamping;
			}

			// store new position and velocity
			thrust::get<0>(t) = make_float4(pos, posData.w);
			thrust::get<1>(t) = make_float4(vel, velData.w);
		}
};
struct my_integrate_functor
{
	float deltaTime;

	__host__ __device__
		my_integrate_functor(float delta_time) : deltaTime(delta_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
			volatile float4 posData = thrust::get<0>(t);
			volatile float4 pos_expData = thrust::get<1>(t);
			volatile float4 velData = thrust::get<2>(t);
			float3 pos = make_float3(posData.x, posData.y, posData.z);
			float3 pos_exp = make_float3(pos_expData.x, pos_expData.y, pos_expData.z);
			float3 vel = make_float3(velData.x, velData.y, velData.z);

			vel += params.gravity * deltaTime;
			vel *= params.globalDamping;

			// new position = old position + velocity * deltaTime
			pos_exp = pos + vel * deltaTime;

			// set this to zero to disable collisions with cube sides
//#if 1

//			if (pos_exp.x > 1.0f - params.particleRadius)
//			{
//				pos_exp.x = 1.0f - params.particleRadius;
//				vel.x *= params.boundaryDamping;
//			}
//
//			if (pos_exp.x < -1.0f + params.particleRadius)
//			{
//				pos_exp.x = -1.0f + params.particleRadius;
//				vel.x *= params.boundaryDamping;
//			}
//
//			/*if (pos_exp.y > 1.0f - params.particleRadius)
//			{
//				pos_exp.y = 1.0f - params.particleRadius;
//				vel.y *= params.boundaryDamping;
//			}*/
//
//			if (pos_exp.z > 1.0f - params.particleRadius)
//			{
//				pos_exp.z = 1.0f - params.particleRadius;
//				vel.z *= params.boundaryDamping;
//			}
//
//			if (pos_exp.z < -1.0f + params.particleRadius)
//			{
//				pos_exp.z = -1.0f + params.particleRadius;
//				vel.z *= params.boundaryDamping;
//			}
//
//#endif
//
//			if (pos_exp.y < -1.0f + params.particleRadius)
//			{
//				pos_exp.y = -1.0f + params.particleRadius;
//				vel.y *= params.boundaryDamping;
//			}


			/*if (pos_exp.x > 2.0f - params.particleRadius)
			{
				pos_exp.x = 2.0f - params.particleRadius;
				vel.x *= params.boundaryDamping;
			}

			if (pos_exp.x < -2.0f + params.particleRadius)
			{
				pos_exp.x = -2.0f + params.particleRadius;
				vel.x *= params.boundaryDamping;
			}

			if (pos_exp.y > 2.0f - params.particleRadius)
			{
				pos_exp.y = 2.0f - params.particleRadius;
				vel.y *= params.boundaryDamping;
			}

			if (pos_exp.z > 2.0f - params.particleRadius)
			{
				pos_exp.z = 2.0f - params.particleRadius;
				vel.z *= params.boundaryDamping;
			}

			if (pos_exp.z < -2.0f + params.particleRadius)
			{
				pos_exp.z = -2.0f + params.particleRadius;
				vel.z *= params.boundaryDamping;
			}*/

//#endif
//
//			if (pos_exp.y < -2.0f + params.particleRadius)
//			{
//				pos_exp.y = -2.0f + params.particleRadius;
//				vel.y *= params.boundaryDamping;
//			}



			// store new position and velocity
			thrust::get<1>(t) = make_float4(pos_exp, posData.w);
			thrust::get<2>(t) = make_float4(vel, velData.w);
		}
};


// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y - 1);
	gridPos.z = gridPos.z & (params.gridSize.z - 1);
	return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

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
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
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




__global__
void reorderDataAndFindCellStartD_sort(uint  *cellStart,
uint  *cellEnd,
float4 *sortedPos,
float4 *sortedPos_exp,
float4 *sortedVel,
float4 *sortedRelative,
float *sortedSDF,
float4 *sortedSDFgradient,
int *sortedPhase,
float *sortedInvMass,
uint  *gridParticleHash,
uint  *gridParticleIndex,
float4 *oldPos,
float4 *oldPos_exp,
float4 *oldVel,
float4 *oldRelative,
float *oldSDF,
float4 *oldSDFgradient,
int *oldPhase,
float *oldInvMass,
uint   numParticles,
uint   numCells)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
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
		float4 pos_exp = FETCH(oldPos_exp, sortedIndex);       // see particles_kernel.cuh
		float4 vel = FETCH(oldVel, sortedIndex);
		float4 relative = FETCH(oldRelative, sortedIndex);
		float sdf = oldSDF[sortedIndex];
		float4 sdf_gradient = FETCH(oldSDFgradient, sortedIndex);
		int phase = oldPhase[sortedIndex];
		float invmass = oldInvMass[sortedIndex];

		sortedPos[index] = pos;
		sortedPos_exp[index] = pos_exp;
		sortedVel[index] = vel;
		sortedRelative[index] = relative;
		sortedSDF[index] = sdf;
		sortedSDFgradient[index] = sdf_gradient;
		sortedPhase[index] = phase;
		sortedInvMass[index] = invmass;

	}


}

















// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
float3 velA, float3 velB,
float radiusA, float radiusB,
float attraction)
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

		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += attraction*relPos;
	}

	return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
uint    index,
float3  pos,
float3  vel,
float4 *oldPos,
float4 *oldVel,
uint   *cellStart,
uint   *cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));
				float3 vel2 = make_float3(FETCH(oldVel, j));

				// collide two spheres
				force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
			}
		}
	}

	return force;
}
__device__
float3 my_collideCell(int3    gridPos,
uint    index,
float3  pos,
float3  vel,
uint *gridParticleIndex,
float4 *oldPos,
float4 *oldVel,
uint   *cellStart,
uint   *cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				uint original_j = gridParticleIndex[j];
				float3 pos2 = make_float3(FETCH(oldPos, original_j));
				float3 vel2 = make_float3(FETCH(oldVel, original_j));

				// collide two spheres
				force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
			}
		}
	}

	return force;
}


//__global__
//void collideD(float4 *newVel,               // output: new velocity
//              float4 *oldPos,               // input: sorted positions
//              float4 *oldVel,               // input: sorted velocities
//              uint   *gridParticleIndex,    // input: sorted particle indices
//              uint   *cellStart,
//              uint   *cellEnd,
//              uint    numParticles)
//{
//    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
//
//    if (index >= numParticles) return;
//
//    // read particle data from sorted arrays
//    float3 pos = make_float3(FETCH(oldPos, index));
//    float3 vel = make_float3(FETCH(oldVel, index));
//
//    // get address in grid
//    int3 gridPos = calcGridPos(pos);
//
//    // examine neighbouring cells
//    float3 force = make_float3(0.0f);
//
//    for (int z=-1; z<=1; z++)
//    {
//        for (int y=-1; y<=1; y++)
//        {
//            for (int x=-1; x<=1; x++)
//            {
//                int3 neighbourPos = gridPos + make_int3(x, y, z);
//                force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
//            }
//        }
//    }
//
//    // collide with cursor sphere
//    force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);
//
//    // write new velocity back to original unsorted location
//    uint originalIndex = gridParticleIndex[index];
//    newVel[originalIndex] = make_float4(vel + force, 0.0f);
//}



//collideD without sorted data
//typedef enum ObjectType{ Particle, Rigid }ObjectType;
__global__
void collideD(float4 *newVel,               // output: new velocity
float4 *oldPos,               // input: unsorted positions
float4 *oldVel,               // input: unsorted velocities
uint   *gridParticleIndex,    // input: sorted particle indices
uint   *cellStart,
uint   *cellEnd,
uint    numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;
	uint originalIndex = gridParticleIndex[index];
	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, originalIndex));
	float3 vel = make_float3(FETCH(oldVel, originalIndex));

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				//force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
				force += my_collideCell(neighbourPos, index, pos, vel, gridParticleIndex, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// collide with cursor sphere
	__syncthreads();
	force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location

	newVel[originalIndex] = make_float4(vel + force, 0.0f);
}
//__global__
//void shaping(
//float *dpos_exp,
//float* dRelative,
//int *dPhase,
//float* dRoate,
//float* dCenter,
//uint* dObjectStart
//)
//{
//	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (index < params.numBodies)
//	{
//		int phase = dPhase[index];
//		uint start = dObjectStart[phase];
//		float3 pos_exp = make_float3(FETCH(dpos_exp, index*4));
//		//set the center to zero
//		if (index >= start && index < start + 3)
//		{
//			dCenter[phase * 4 + index - start] = 0;
//		}
//		atomicAdd(&dCenter[phase * 4], (float)pos_exp.x);
//	}
//	
//}
//__global__
//void Cal_Obj_Center(
//float *dpos_exp,
//int *dPhase,
//float* dCenter,
//uint* dObjectStart,
//uint* dObjectSize
//)
//{
//	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (index < params.numBodies)
//	{
//		int phase = dPhase[index];
//		uint start = dObjectStart[phase];
//		float3 pos_exp = make_float3(FETCH(dpos_exp, index * 4));
//		//set the center to zero
//		if (index >= start && index < start + 3)
//		{
//			dCenter[phase * 4 + index - start] = 0;
//		}
//		atomicAdd(&dCenter[phase * 4], (float)pos_exp.x);
//	}
//
//}
	__global__ 
		void Float4ToFloat(int size, float4 *dF4, float *dFx, float *dFy, float *dFz, float *dFw)
		{
			uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
			if (index < size)
			{
				dFx[index] = dF4[index].x;
				dFy[index] = dF4[index].y;
				dFz[index] = dF4[index].z;
				dFw[index] = dF4[index].w;
			}
		}
	__device__
		void Vector_Mul(float* a, float* b, float* result ,int size)
		{
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < size; j++)
				{
					result[size * i + j] = a[i] * b[j];
				}
			}

		}
	__device__ 
	void MatrixMul(float* Ma, float* Mb, float* Result, int size1, int size2, int size3)
	{
		int index;
		for (int row = 0; row < size1; row++)
		{
			for (int col = 0; col < size3; col++)
			{
				Result[size3*row + col] = 0;
				for (int i = 0; i < size2; i++)
				{
					Result[size3*row + col] += Ma[row * size2 + i] * Mb[i * size3 + col];
				}
			}
		}
		return;
	}
	__global__
		void Cal_Matrix_A_element_kernel(int size, float4 center, float* dPos_exp, float* dRelative, float* dTempA )
		{
			uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
			if (index < size)
			{
				float3 x_exp = make_float3(
					dPos_exp[index * 4],
					dPos_exp[index * 4 + 1],
					dPos_exp[index * 4 + 2]);
				float3 r = make_float3(
					dRelative[index * 4],
					dRelative[index * 4 + 1],
					dRelative[index * 4 + 2]);
				float3 c = make_float3(center.x, center.y, center.z);
				float3 new_r = x_exp - c;
				Vector_Mul((float*)&new_r, (float*)&r, &dTempA[index * 3 * 3], 3);
			}
		}
	//template <int BLOCK_SIZE, int DATA_STEP>
	__global__ 
		void Cal_Matrix_A_element_BR_kernal(float* dPos_exp, float* dCenter, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType,float* result, int num_Particle)
		{
			__shared__ float3 c;
			__shared__ uint start;// = dObjectStart[index_obj];
			__shared__ uint size;// = dObjectSize[index_obj];
			int index_obj = blockIdx.x;
			int index_th = threadIdx.x;
			int block_size = blockDim.x;
			
			ObjectType type;
			if (index_th == 0)
			{
				c.x = dCenter[4 * index_obj];
				c.y = dCenter[4 * index_obj + 1];
				c.z = dCenter[4 * index_obj + 2];
				size = dObjectSize[index_obj];
				start = dObjectStart[index_obj];
				type = dObjectType[index_obj];
			}
			__syncthreads();
			if (type < Rigid || type > Deformation_quad_plasticity)
				return;
			float*  data_pos = dPos_exp + start * 4;
			float*  data_relative = dRelative + start * 4;
			float* data_result = result + 3 * 3 * start;
			int num_remain = size;
			int i = 0;
			int index;
			while (num_remain > 0)
			{
				index = index_th + i * block_size;
				if (index < size)
				{
					float3 x_exp = make_float3(
						data_pos[index * 4],
						data_pos[index * 4 + 1],
						data_pos[index * 4 + 2]);
					float3 r = make_float3(
						data_relative[index * 4],
						data_relative[index * 4 + 1],
						data_relative[index * 4 + 2]);
					float3 new_r = x_exp - c;
					Vector_Mul((float*)&new_r, (float*)&r, &data_result[index * 3 * 3], 3);
				}
				num_remain -= block_size;
				i++;
			}
		}

	__global__
	void Cal_Matrix_A_quad_element_kernel(int size, float4 center, float* dPos_exp, float* dRelative, float* dTempA)
	{
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index < size)
		{
			float3 x_exp = make_float3(
				dPos_exp[index * 4],
				dPos_exp[index * 4 + 1],
				dPos_exp[index * 4 + 2]);
			float3 r = make_float3(
				dRelative[index * 4],
				dRelative[index * 4 + 1],
				dRelative[index * 4 + 2]);
			float3 c = make_float3(center.x, center.y, center.z);
			float3 new_r = x_exp - c;
			float q_quad[9];
			q_quad[0] = r.x; q_quad[1] = r.y; q_quad[2] = r.z;
			q_quad[3] = r.x * r.x; q_quad[4] = r.y * r.y; q_quad[5] = r.z * r.z;
			q_quad[6] = r.x * r.y; q_quad[7] = r.y * r.z; q_quad[8] = r.z * r.x;
			//Vector_Mul((float*)&new_r, (float*)&r, &dTempA[index * 3 * 3], 3);
			MatrixMul((float*)&new_r, q_quad, &dTempA[index * 3 * 9], 3 ,1 , 9);
		}
	}
	__global__
		void Cal_Matrix_A_quad_element_BR_kernel(float* dPos_exp, float* dCenter, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType, float* result, int num_Particle)
	{
			__shared__ float3 c;
			__shared__ uint start;// = dObjectStart[index_obj];
			__shared__ uint size;// = dObjectSize[index_obj];
			int index_obj = blockIdx.x;
			int index_th = threadIdx.x;
			int block_size = blockDim.x;

			ObjectType type;
			if (index_th == 0)
			{
				c.x = dCenter[4 * index_obj];
				c.y = dCenter[4 * index_obj + 1];
				c.z = dCenter[4 * index_obj + 2];
				size = dObjectSize[index_obj];
				start = dObjectStart[index_obj];
				type = dObjectType[index_obj];
			}
			__syncthreads();
			if (type < Deformation_quad || type > Deformation_quad_plasticity)
				return;
			float*  data_pos = dPos_exp + start * 4;
			float*  data_relative = dRelative + start * 4;
			float* data_result = result + 3 * 9 * start;
			
			int num_remain = size;
			int i = 0;
			int index;
			while (num_remain > 0)
			{
				index = index_th + i * block_size;
				if (index < size)
				{
					float3 x_exp = make_float3(
						data_pos[index * 4],
						data_pos[index * 4 + 1],
						data_pos[index * 4 + 2]);
					float3 r = make_float3(
						data_relative[index * 4],
						data_relative[index * 4 + 1],
						data_relative[index * 4 + 2]);
					float3 new_r = x_exp - c;
					float q_quad[9];
					q_quad[0] = r.x; q_quad[1] = r.y; q_quad[2] = r.z;
					q_quad[3] = r.x * r.x; q_quad[4] = r.y * r.y; q_quad[5] = r.z * r.z;
					q_quad[6] = r.x * r.y; q_quad[7] = r.y * r.z; q_quad[8] = r.z * r.x;
					//Vector_Mul((float*)&new_r, (float*)&r, &data_result[index * 3 * 3], 3);
					MatrixMul((float*)&new_r, q_quad, &data_result[index * 3 * 9], 3, 1, 9);
				}
				num_remain -= block_size;
				i++;
			}
		}

	__global__
		void MatrixToFloat(float* dMatrix, float* dFloat, int i, int j,int size,int n_row, int n_col)
		{
			uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
			if (index < size)
				dFloat[index] = dMatrix[n_row * n_col * index + n_col * i + j];
		}
	__device__
		float3 MatrixMulVector(float* Matrix, float* Vector, int size)
		{
			float result[3];
			for (int i = 0; i < 3; i++)
			{
				result[i] = 0;
				for (int j = 0; j < 3; j++)
				{
					result[i] += Matrix[i * 3 + j] * Vector[j];
				}
			}
			return *((float3*)result);
		}
	__device__
		float3 MatrixMulVector_quad(float* Matrix, float* Vector)
		{
			float result[3];
			for (int i = 0; i < 3; i++)
			{
				result[i] = 0;
				for (int j = 0; j < 9; j++)
				{
					result[i] += Matrix[i * 9 + j] * Vector[j];
				}
			}
			return *((float3*)result);

		}
	
	__global__
		void Shaping_Cal_pos_kernel(float *dPos_exp, float* dRelative, int *dPhase, float* dCenter, float *dRoate,float* dRoate_quad, ObjectType *dObjectType,uint num_Particles)
		{
			uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
			if (index < num_Particles)
			{
				
				int index_object = dPhase[index];
				ObjectType type = dObjectType[index_object];
				if (type >= Rigid && type <= Deformation_quad_plasticity)
				{
					float3 x_e = make_float3(
						dPos_exp[index * 4],
						dPos_exp[index * 4 + 1],
						dPos_exp[index * 4 + 2]
						);
					float3 r = make_float3(
						dRelative[index * 4],
						dRelative[index * 4 + 1],
						dRelative[index * 4 + 2]
						);
					float3 c = make_float3(
						dCenter[index_object * 4],
						dCenter[index_object * 4 + 1],
						dCenter[index_object * 4 + 2]
						);
					float3 delta_x;
					if (type >= Rigid && type <= Deformation_linear_plasticity)
					{
						float Roate[3 * 3];
						memcpy(Roate, dRoate + index_object * 3 * 3, 3 * 3 * sizeof(float));
						delta_x = MatrixMulVector(Roate, (float*)&r, 3) + c - x_e;
					}
					else if (type >= Deformation_quad && type <= Deformation_quad_plasticity)
					{
						float Roate[3 * 9];
						float q_quad[9];
						q_quad[0] = r.x; q_quad[1] = r.y; q_quad[2] = r.z;
						q_quad[3] = r.x * r.x; q_quad[4] = r.y * r.y; q_quad[5] = r.z * r.z;
						q_quad[6] = r.x * r.y; q_quad[7] = r.y * r.z; q_quad[8] = r.z * r.x;
						memcpy(Roate, dRoate_quad + index_object * 3 * 9, 3 * 9 * sizeof(float));
						delta_x = MatrixMulVector_quad(Roate, q_quad) + c - x_e;
					}
					
					__syncthreads();
					dPos_exp[index * 4] += params.Omega * delta_x.x / 4;
					dPos_exp[index * 4 + 1] += params.Omega * delta_x.y / 4;
					dPos_exp[index * 4 + 2] += params.Omega * delta_x.z / 4;
					/*dPos_exp[index * 4] += delta_x.x;
					dPos_exp[index * 4 + 1] += delta_x.y;
					dPos_exp[index * 4 + 2] += delta_x.z ;*/
				}
				
			}
		}
	/*__global__ void Update_Mesh(ParticleMesh , float* dRoate, float* dRoate_quad)
	{
		

	}*/
	__global__ void Update_Vertex(float* dVertex, float* dNormal, float *dRelative, float* dRoate, float* dRoate_quad, float* dcenter, int num, ObjectType type)
	{
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index < num)
		{
			float Roate[3 * 3];
			float center[3];
			//float3 vertex = make_float3(dVertex[index * 3], dVertex[index * 3 + 1], dVertex[index * 3 + 2]);
			
			float3 r = make_float3(dRelative[index * 3], dRelative[index * 3 + 1], dRelative[index * 3 + 2]);
			memcpy(Roate, dRoate, 3 * 3 * sizeof(float));
			memcpy(center, dcenter, 3 * sizeof(float));
			float3 c = make_float3(center[0], center[1], center[2]);
			float3 vertex_new;
			float3 new_r = MatrixMulVector(Roate, (float*)&r, 3);
			vertex_new = new_r + c;
			dVertex[index * 3] = vertex_new.x;
			dVertex[index * 3 + 1] = vertex_new.y;
			dVertex[index * 3 + 2] = vertex_new.z;
			/*if (type == Rigid)
			{
				float3 normal = make_float3(dNormal[index * 3], dNormal[index * 3 + 1], dNormal[index * 3 + 2]);
				float3 tmp;
				tmp = r + normal;

				tmp = MatrixMulVector(Roate, (float*)&tmp, 3);
				normal = tmp - new_r;
				dNormal[index * 3] = normal.x;
				dNormal[index * 3 + 1] = normal.y;
				dNormal[index * 3 + 2] = normal.z;
			}*/
			
			

			
		} 
	}

	__global__ void Update_VertexFullPal(float* dVertex, float* dNormal, float* Normal_Original, float *dRelative, int *dPhaseVertex, ObjectType *dObjectType, float* dRoate, float* dRoate_quad, float* dcenter, int num)
	{
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index < num)
		{
			float Roate[3 * 3];
			float Roate_quad[3 * 9];
			float center[3];
			int index_obj = dPhaseVertex[index];
			//float3 vertex = make_float3(dVertex[index * 3], dVertex[index * 3 + 1], dVertex[index * 3 + 2]);
			float3 new_r;
			float3 r = make_float3(dRelative[index * 3], dRelative[index * 3 + 1], dRelative[index * 3 + 2]);
			memcpy(center, dcenter + index_obj * 4, 3 * sizeof(float));
			ObjectType type = dObjectType[index_obj];
			float3 c = make_float3(center[0], center[1], center[2]);
			float3 vertex_new;
			if (type >= Rigid && type <= Deformation_linear_plasticity)
			{
				memcpy(Roate, dRoate + index_obj * 3 * 3, 3 * 3 * sizeof(float));
				new_r = MatrixMulVector(Roate, (float*)&r, 3);
				vertex_new = new_r + c;
				dVertex[index * 3] = vertex_new.x;
				dVertex[index * 3 + 1] = vertex_new.y;
				dVertex[index * 3 + 2] = vertex_new.z;
			}
			else if(type >= Deformation_quad && type <= Deformation_quad_plasticity)
			{
				memcpy(Roate_quad, dRoate_quad + index_obj * 3 * 9, 3 * 9 * sizeof(float));
				float q_quad[9];
				q_quad[0] = r.x; q_quad[1] = r.y; q_quad[2] = r.z;
				q_quad[3] = r.x * r.x; q_quad[4] = r.y * r.y; q_quad[5] = r.z * r.z;
				q_quad[6] = r.x * r.y; q_quad[7] = r.y * r.z; q_quad[8] = r.z * r.x;
				new_r = MatrixMulVector(Roate, (float*)&r, 3);
				//memcpy(Roate, dRoate_quad + index_object * 3 * 9, 3 * 9 * sizeof(float));
				new_r = MatrixMulVector_quad(Roate_quad, q_quad);


				vertex_new = new_r + c;
				dVertex[index * 3] = vertex_new.x;
				dVertex[index * 3 + 1] = vertex_new.y;
				dVertex[index * 3 + 2] = vertex_new.z;
			}
			
			if (type == Rigid)
			{
				float3 normal = make_float3(Normal_Original[index * 3], Normal_Original[index * 3 + 1], Normal_Original[index * 3 + 2]);
				float3 tmp;
				tmp = r + normal;

				tmp = MatrixMulVector(Roate, (float*)&tmp, 3);
				normal = tmp - new_r;
				dNormal[index * 3] = normal.x;
				dNormal[index * 3 + 1] = normal.y;
				dNormal[index * 3 + 2] = normal.z;
			}




		}
	}

	__global__ void Update_Normal(float* dVertex, uint* dFace, float* dNormal, int * dPhaseFace, int* dPhaseVertex,ObjectType *dObjectType, int num_face, int num_vertex)
	{
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index < num_vertex)
		{
			int index_obj = dPhaseVertex[index];
			ObjectType type = dObjectType[index_obj];
			if (type != Rigid)
			{
				dNormal[index * 3] = 0;
				dNormal[index * 3 + 1] = 0;
				dNormal[index * 3 + 2] = 0;
			}
			

			//num_tri_per_point[index] = 0;
		}
		__syncthreads();
		if (index < num_face)
		{
			int index_obj = dPhaseFace[index];
			ObjectType type = dObjectType[index_obj];
			if (type == Rigid)
				return;
			float3 n;
			uint i_1 = dFace[index * 3];
			uint i_2 = dFace[index * 3 + 1];
			uint i_3 = dFace[index * 3 + 2];
			float3 v1 = make_float3(dVertex[i_1 * 3], dVertex[i_1 * 3 + 1], dVertex[i_1 * 3 + 2]);
			float3 v2 = make_float3(dVertex[i_2 * 3], dVertex[i_2 * 3 + 1], dVertex[i_2 * 3 + 2]);
			float3 v3 = make_float3(dVertex[i_3 * 3], dVertex[i_3 * 3 + 1], dVertex[i_3 * 3 + 2]);
			
			float3 e12 = v2 - v1;
			float3 e13 = v3 - v1;
			n = cross(e12, e13);
			n /= length(n);
			atomicAdd(&(dNormal[i_1 * 3]), n.x);
 			atomicAdd(&dNormal[i_1 * 3 + 1], n.y);
			atomicAdd(&dNormal[i_1 * 3 + 2], n.z);
			
			//atomicAdd(&num_tri_per_point[i_1], 1);

			float3 e21 = -e12;
			float3 e23 = v3 - v2;
			n = cross(e23, e21);
			n /= length(n);
			atomicAdd(&dNormal[i_2 * 3], n.x);
			atomicAdd(&dNormal[i_2 * 3 + 1], n.y);
			atomicAdd(&dNormal[i_2 * 3 + 2], n.z);
			
			//atomicAdd(&num_tri_per_point[i_2], 1);

			float3 e31 = -e13;
			float3 e32 = -e23;
			n = cross(e31, e32);
			n /= length(n);
			atomicAdd(&dNormal[i_3 * 3], n.x);
			atomicAdd(&dNormal[i_3 * 3 + 1], n.y);
			atomicAdd(&dNormal[i_3 * 3 + 2], n.z);

			//atomicAdd(&num_tri_per_point[i_3], 1);

		}
		__syncthreads();
		
	}
	__global__ void Normalize_Normal(float* dNormal, int* dPhaseVertex, ObjectType* dObjectType, int num_vertex)
	{
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

		if (index < num_vertex)
		{
			int index_obj = dPhaseVertex[index];
			ObjectType type = dObjectType[index_obj];
			if (type == Rigid)
				return;
			float3 n = make_float3(dNormal[index * 3], dNormal[index * 3 + 1], dNormal[index * 3 + 2]);
			n /= length(n);
			dNormal[index * 3] = n.x;
			dNormal[index * 3 + 1] = n.y;
			dNormal[index * 3 + 2] = n.z;
		}
	}
	__global__
		void Update_Change_kernel(float *dPos, float *dPos_exp,  float *dVel, int numParticles, float delta_time,float epsilon)
		{
			uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

			if (index < numParticles)
			{
				float4 pos_exp, vel, pos;
				float4 *dPos_f4, *dPos_exp_f4, *dVel_f4;
				dPos_f4 = (float4*)dPos;
				dPos_exp_f4 = (float4*)dPos_exp;
				dVel_f4 = (float4*)dVel;

				pos_exp = dPos_exp_f4[index];
				pos = dPos_f4[index];

				dVel_f4[index] = (pos_exp - pos) / delta_time;
				if (length(pos_exp - pos) > epsilon)
				{
					dPos_f4[index] = pos_exp;
				}
					
			}
		}
	__device__ float3 CalNorm(uint    index,
		uint	original_i,
		uint	original_j,
		float3 posA, float3 posB,
		float3 velA, float3 velB,
		float radiusA, float radiusB,
		float invmA, float invmB,
		int phaseA,
		int phaseB,
		float3 relative_A, float3 relative_B,
		ObjectType TypeA,
		ObjectType TypeB,
		float* dSDF,
		float4*	dSDF_gridiant,
		float* dRoate,
		float* d
		)
	{
		float3 relPos = posB - posA;

		float dist = length(relPos);
		float collideDist = radiusA + radiusB;
		float ra[9], rb[9];
		float3 norm;
		memcpy(ra, dRoate + phaseA * 3 * 3, 9 * sizeof(float));
		memcpy(rb, dRoate + phaseB * 3 * 3, 9 * sizeof(float));
		float3 delta_x = make_float3(0.0f);
		if (dist < collideDist*1)
		{
			//float d;
			if (TypeA == Rigid)
			{
				float sdfA, sdfB;
				float3 sdf_gridiantA = make_float3(FETCH(dSDF_gridiant, original_i));
				float3 sdf_gridiantB = make_float3(FETCH(dSDF_gridiant, original_j));
				sdfA = dSDF[original_i];
				sdf_gridiantA = make_float3(FETCH(dSDF_gridiant, original_i));
				*d = abs(sdfA);
				float3 tmp;
				tmp = relative_A + sdf_gridiantA;
				norm = MatrixMulVector(ra, (float*)&tmp, 3) - MatrixMulVector(ra, (float*)&relative_A, 3);
				if (TypeB == Rigid)
				{
					sdfB = dSDF[original_j];
					sdf_gridiantB = make_float3(FETCH(dSDF_gridiant, original_j));
					if (abs(sdfA) > abs(sdfB))
					{
						*d = abs(sdfB);
						tmp = relative_B + sdf_gridiantB;
						norm = MatrixMulVector(rb, (float*)&tmp, 3) - MatrixMulVector(rb, (float*)&relative_B, 3);
						norm *= -1;
					}
				}

				if (abs(sdfA) < 1.1 * collideDist)
				{
					if (dot(relPos, norm) < 0)
					{
						norm = relPos - 2 * dot(relPos, norm)*norm;
					}
					else
					{
						norm = relPos;
					}
					
					*d = abs((length(relPos) - collideDist));
				}
				norm /= length(norm);
			}
			else
			{
				norm = relPos / length(relPos);
				*d = collideDist - length(relPos);
			}
		}
		return norm;
	}

	__device__ float3 CalNorm_sort(
		uint	i,
		uint	j,
		float3 posA, float3 posB,
		float radiusA, float radiusB,
		float invmA, float invmB,
		int phaseA,
		int phaseB,
		float3 relative_A, float3 relative_B,
		ObjectType TypeA,
		ObjectType TypeB,
		float* SortedSDF,
		float4*	SortedSDF_gridiant,
		float* dRoate,
		float* d
		)
	{
		float3 relPos = posB - posA;

		float dist = length(relPos);
		float collideDist = radiusA + radiusB;
		float ra[9], rb[9];
		float3 norm;
		memcpy(ra, dRoate + phaseA * 3 * 3, 9 * sizeof(float));
		memcpy(rb, dRoate + phaseB * 3 * 3, 9 * sizeof(float));
		float3 delta_x = make_float3(0.0f);
		if (dist < collideDist * 1)
		{
			//float d;
			if (TypeA == Rigid)
			{
				float sdfA, sdfB;
				float3 sdf_gridiantA = make_float3(FETCH(SortedSDF_gridiant, i));
				float3 sdf_gridiantB = make_float3(FETCH(SortedSDF_gridiant, j));
				sdfA = SortedSDF[i];
				sdf_gridiantA = make_float3(FETCH(SortedSDF_gridiant, i));
				*d = abs(sdfA);
				float3 tmp;
				tmp = relative_A + sdf_gridiantA;
				norm = MatrixMulVector(ra, (float*)&tmp, 3) - MatrixMulVector(ra, (float*)&relative_A, 3);
				if (TypeB == Rigid)
				{
					sdfB = SortedSDF[j];
					sdf_gridiantB = make_float3(FETCH(SortedSDF_gridiant, j));
					if (abs(sdfA) > abs(sdfB))
					{
						*d = abs(sdfB);
						tmp = relative_B + sdf_gridiantB;
						norm = MatrixMulVector(rb, (float*)&tmp, 3) - MatrixMulVector(rb, (float*)&relative_B, 3);
						norm *= -1;
					}
				}

				if (abs(sdfA) < 1.1 * collideDist)
				{
					if (dot(relPos, norm) < 0)
					{
						norm = relPos - 2 * dot(relPos, norm)*norm;
					}
					else
					{
						norm = relPos;
					}

					*d = abs((length(relPos) - collideDist));
				}
				norm /= length(norm);
			}
			else
			{
				norm = relPos / length(relPos);
				*d = collideDist - length(relPos);
			}
		}
		return norm;
	}
	__device__ float3 non_penetration(float3 norm, float3 pos_exp_A, float3 pos_exp_B, float radiusA, float radiusB, float invmassA, float invmassB)
	{
		float3 delta_x;
		float3 x_AB = pos_exp_B - pos_exp_A;
		delta_x = (length(x_AB) - radiusA - radiusB)*x_AB / length(x_AB);
		delta_x *= -(invmassA / (invmassA + invmassB));
		return delta_x;
	}
	__device__ float min_float(float a, float b)
	{
		if (a > b)
			return b;
		else
			return a;
	}
	__device__ float3 frictionSphere(float invmassA, float invmassB, float3 pos_exp_A, float3 pos_exp_B, float3 pos_old_A, float3 pos_old_B, float radiusA, float radiusB)
	{
		float factor_fri_static = params.factor_s;
		float factor_fri_kinetic = params.factor_k;
		float d;
		float3 delta_x;
		float3 delta_x_tangential;
		float3 delta_x_non_penetration;
		float3 translationA = pos_exp_A - pos_old_A;
		float3 translationB = pos_exp_B - pos_old_B;
		float3 rel_trans = translationA - translationB;
		float3 norm = (pos_exp_B - pos_exp_A) / length(pos_exp_B - pos_exp_A);
		for (int i = 0; i < 1; i++)
		{
			delta_x_tangential = rel_trans - norm*(dot(norm, rel_trans));
			delta_x_non_penetration = non_penetration(norm, pos_exp_A, pos_exp_B, radiusA, radiusB, invmassA, invmassB);
			d = abs(dot(delta_x_non_penetration, norm));
			if (length(delta_x_tangential) < 1e-4)
				delta_x = make_float3(0);
			else if (length(delta_x_tangential) < factor_fri_static*d)
			{
				delta_x = delta_x_tangential;
			}
			else
			{
				delta_x = delta_x_tangential * min_float(factor_fri_kinetic * d / length(delta_x_tangential), 1);
			}
			delta_x *= -(invmassA / (invmassA + invmassB));
			if (length(delta_x) != 0)
				delta_x += 0;
		}
		
		return delta_x;
	}
	
	__device__
		float3 collideSpheres_SDF(float3 posA, float3 posB,
		float3 velA, float3 velB,
		float radiusA, float radiusB,
		float sdfA, float sdfB,
		float3 sdf_gridiantA, float3 sdf_gridiantB,
		float invmA, float invmB,
		float* RoateA,
		float* RoateB,
		int* n_collision
		)
	{
			// calculate relative position
			float3 relPos = posB - posA;

			float dist = length(relPos);
			float collideDist = radiusA + radiusB;
			float ra[9], rb[9];
			memcpy(ra, RoateA, 9 * sizeof(float));
			memcpy(rb, RoateB, 9 * sizeof(float));
			float3 delta_x = make_float3(0.0f);

			if (dist < collideDist*0.95)
			{
				float d;
				float3 norm;
				(*n_collision)++;
				if (abs(sdfA) < abs(sdfB))
				{
					d = abs(sdfA);
					norm = MatrixMulVector(RoateA, (float*)&sdf_gridiantA, 3);
				}
				else
				{
					d = abs(sdfB);
					norm = -MatrixMulVector(RoateB, (float*)&sdf_gridiantB, 3);
				}
				if (abs(sdfA) < 1.1 * radiusA)
				{
					if (dot(relPos, norm) < 0)
					{
						norm = relPos - 2 * dot(relPos, norm)*norm;
					}
					else
					{
						norm = relPos;
					}
					norm /= length(norm);
					d = abs((length(relPos) - radiusA - radiusB));
				}
				delta_x = -(invmA / (invmA + invmB))*(d * norm);
			}

			return delta_x;
		}
	__device__ float stack_height(float3 pos,float ground_y)
	{
		return pos.y - ground_y;
	}
	__device__ float mass_scale(float3 pos,int k)
	{
		return exp(-k*stack_height(pos, -1));
	}
	__device__ float3 MyCollideSpheres(float invmassA,float invmassB, float d, float3 norm)
	{
		return -(invmassA / (invmassA + invmassB))*(d * norm);
	}
	__device__ float3 MyContactCell(int3    gridPos,
		uint    index,
		uint	original_i,
		float3  pos,
		float3  vel,
		uint *gridParticleIndex,
		float4 *oldPos,
		float4 *Pos_exp,
		float4 *oldVel,
		float4 *dRelative,
		uint   *cellStart,
		uint   *cellEnd,
		float*	dInvMass,
		float* dSDF,
		int* dPhase,
		float4*	dSDF_gridiant,
		float* dRoate,
		ObjectType* dObjectType,
		int *n_collision
		)
	{
		uint gridHash = calcGridHash(gridPos);
		
		// get start of bucket for this cell
		uint startIndex = FETCH(cellStart, gridHash);
		float3 delta_x = make_float3(0.0f);
		int k = params.k_mass_scale;
		if (startIndex != 0xffffffff)          // cell is not empty
		{
			// iterate over particles in this cell
			uint endIndex = FETCH(cellEnd, gridHash);

			for (uint j = startIndex; j<endIndex; j++)
			{
				if (j != index)                // check not colliding with self
				{
					uint original_j = gridParticleIndex[j];
					float3 pos2 = make_float3(FETCH(Pos_exp, original_j));
					float3 vel2 = make_float3(FETCH(oldVel, original_j));
					int phaseA = dPhase[original_i];
					int phaseB = dPhase[original_j];
					
					/*memcpy(RA, dRoate + phaseA * 3 * 3, 3 * 3 * sizeof(float));
					memcpy(RB, dRoate + phaseB * 3 * 3, 3 * 3 * sizeof(float));*/
					// collide two spheres
					//force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
					if (length(pos - pos2) < params.particleRadius * 2)
					{
						float3 norm;
						float d;
						float d_friction;
						n_collision++;
						if ((dObjectType[phaseA] >= Rigid &&  dObjectType[phaseA] <=  Deformation_quad_plasticity) && phaseA == phaseB)
							continue;
						//if (phaseA != phaseB)
						else
						{
							float InvMassA = dInvMass[original_i] / mass_scale(pos, k);
							float InvMassB = dInvMass[original_j] / mass_scale(pos2, k);
							float3 pos_oldA = make_float3(FETCH(oldPos, original_i));
							float3 pos_oldB = make_float3(FETCH(oldPos, original_j));
							float3 relative_A = make_float3(FETCH(dRelative, original_i));
							float3 relative_B = make_float3(FETCH(dRelative, original_j));

							float RA[3 * 3], RB[3 * 3];
							ObjectType TypeA = dObjectType[phaseA];
							ObjectType TypeB = dObjectType[phaseB];
							norm = CalNorm(index, original_i, original_j, pos, pos2, vel, vel2,
								params.particleRadius, params.particleRadius, InvMassA, InvMassB,
								phaseA, phaseB, relative_A,relative_B , TypeA, TypeB, dSDF, dSDF_gridiant, dRoate, &d);
							delta_x += MyCollideSpheres(InvMassA, InvMassB, d, norm);
							if (params.friction)
								delta_x += frictionSphere(InvMassA, InvMassB, pos, pos2, pos_oldA, pos_oldB, params.particleRadius, params.particleRadius);
						}
					}
				}
			}
		}
		return delta_x;
	}





	__device__ float3 MyContactCell_sort(int3    gridPos,
		uint    index,
		float3  pos,
		uint *gridParticleIndex,
		float4 *SortedPos,
		float4 *SortedPos_exp,
		float4 *SortedRelative,
		float*	SortedInvMass,
		float* SortedSDF,
		int* SortedPhase,
		float4*	SortedSDF_gridiant,
		uint   *cellStart,
		uint   *cellEnd,
		float* dRoate,
		ObjectType* dObjectType,
		int *n_collision
		)
	{
		uint gridHash = calcGridHash(gridPos);

		// get start of bucket for this cell
		uint startIndex = FETCH(cellStart, gridHash);
		float3 delta_x = make_float3(0.0f);
		int k = params.k_mass_scale;
		if (startIndex != 0xffffffff)          // cell is not empty
		{
			// iterate over particles in this cell
			uint endIndex = FETCH(cellEnd, gridHash);

			for (uint j = startIndex; j<endIndex; j++)
			{
				if (j != index)                // check not colliding with self
				{
					//uint original_j = gridParticleIndex[j];
					float3 pos2 = make_float3(FETCH(SortedPos_exp, j));
					//float3 vel2 = make_float3(FETCH(SortedVel, j));
					int phaseA = SortedPhase[index];
					int phaseB = SortedPhase[j];
					
					// collide two spheres
					//force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
					if (length(pos - pos2) < params.particleRadius * 2)
					{
						float3 norm;
						float d;
						float d_friction;
						n_collision++;
						if ((dObjectType[phaseA] >= Rigid &&  dObjectType[phaseA] <= Deformation_quad_plasticity) && phaseA == phaseB)
							continue;
						//if (phaseA != phaseB)
						else
						{
							float3 pos_oldA = make_float3(FETCH(SortedPos, index));
							float3 pos_oldB = make_float3(FETCH(SortedPos, j));
							float InvMassA = SortedInvMass[index] / mass_scale(pos, k);
							float InvMassB = SortedInvMass[j] / mass_scale(pos2, k);
							float3 relative_A = make_float3(FETCH(SortedRelative, index));
							float3 relative_B = make_float3(FETCH(SortedRelative, j));

							float RA[3 * 3], RB[3 * 3];
							ObjectType TypeA = dObjectType[phaseA];
							ObjectType TypeB = dObjectType[phaseB];
							/*memcpy(RA, dRoate + phaseA * 3 * 3, 3 * 3 * sizeof(float));
							memcpy(RB, dRoate + phaseB * 3 * 3, 3 * 3 * sizeof(float));*/
							norm = CalNorm_sort(index, j, pos, pos2, params.particleRadius, params.particleRadius, 
								InvMassA, InvMassB, phaseA, phaseB, relative_A, relative_B, 
								TypeA, TypeB, SortedSDF, SortedSDF_gridiant, dRoate, &d);
							delta_x += MyCollideSpheres(InvMassA, InvMassB, d, norm);
							if (params.friction)
								delta_x += frictionSphere(InvMassA, InvMassB, pos, pos2, pos_oldA, pos_oldB, params.particleRadius, params.particleRadius);
						}
					}
				}
			}
		}
		return delta_x;
	}




	__device__
		float3 ContactCell(int3    gridPos,
		uint    index,
		uint	original_i,
		float3  pos,
		float3  vel,
		uint *gridParticleIndex,
		float4 *oldPos,
		float4 *oldVel,
		uint   *cellStart,
		uint   *cellEnd,
		float*	dInvMass,
		float* dSDF,
		int* dPhase,
		float4*	dSDF_gridiant,
		float* dRoate,
		int *n_collision,
		int k
		)
	{
			uint gridHash = calcGridHash(gridPos);

			// get start of bucket for this cell
			uint startIndex = FETCH(cellStart, gridHash);
			float3 delta_x = make_float3(0.0f);

			if (startIndex != 0xffffffff)          // cell is not empty
			{
				// iterate over particles in this cell
				uint endIndex = FETCH(cellEnd, gridHash);

				for (uint j = startIndex; j<endIndex; j++)
				{
					if (j != index)                // check not colliding with self
					{
						uint original_j = gridParticleIndex[j];
						float3 pos2 = make_float3(FETCH(oldPos, original_j));
						float3 vel2 = make_float3(FETCH(oldVel, original_j));
						float InvMassA = dInvMass[original_i]/mass_scale(pos,k);
						float InvMassB = dInvMass[original_j]/mass_scale(pos2,k);
						float sdfA = dSDF[original_i];
						float sdfB = dSDF[original_j];
						float3 sdf_gridiantA = make_float3(FETCH(dSDF_gridiant, original_i));
						float3 sdf_gridiantB = make_float3(FETCH(dSDF_gridiant, original_j));
						int phaseA = dPhase[original_i];
						int phaseB = dPhase[original_j];
						float RA[3*3],RB[3*3];
						memcpy(RA, dRoate + phaseA * 3 * 3 , 3 * 3 * sizeof(float));
						memcpy(RB, dRoate + phaseB * 3 * 3 , 3 * 3 * sizeof(float));
						// collide two spheres
						//force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
						if (length(pos - pos2) < params.particleRadius * 2)
						{
							// 
							if (phaseA != phaseB &&length(pos - pos2) < params.particleRadius * 2)
							{
								delta_x += collideSpheres_SDF(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius,
									sdfA, sdfB, sdf_gridiantA, sdf_gridiantB, InvMassA, InvMassB,
									RA, RB, n_collision);
							}
						}
						
					}
				}
			}

			return delta_x;
		}
	__global__
		void contactD(float4 *newVel,               // output: new velocity
		float4 *oldPos,               // input: unsorted positions
		float4 *Pos_exp,
		float4 *oldVel,            // input: unsorted velocities
		float4* dRelative,
		uint   *gridParticleIndex,    // input: sorted particle indices
		uint   *cellStart,
		uint   *cellEnd,
		float*	dInvMass,
		float* dSDF,
		int* dPhase,
		float4*	dSDF_gridiant,
		float* dRoate,
		ObjectType* dObjectType,
		uint    numParticles,
		float delta_time,
		IterationType IterType
		)
	{
			uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
			
			if (index >= numParticles) return;
			uint originalIndex = gridParticleIndex[index];
			if (originalIndex >= 27)
				int a = index;
			// read particle data from sorted arrays
			
			float3 pos;
			if (IterType == Pre_Stablization)
				pos = make_float3(FETCH(oldPos, originalIndex));
			else
				pos = make_float3(FETCH(Pos_exp, originalIndex));
			float3 pos_exp = make_float3(FETCH(Pos_exp, originalIndex));
			float3 vel = make_float3(FETCH(oldVel, originalIndex));

			// get address in grid
			int3 gridPos = calcGridPos(pos_exp);

			// examine neighbouring cells
			float3 delta_x = make_float3(0.0f);
			int n_collision = 0;
			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbourPos = gridPos + make_int3(x, y, z);
					
						/*if (IterType == Pre_Stablization)
							delta_x += ContactCell(neighbourPos, index, originalIndex, pos, vel,
								gridParticleIndex, oldPos, oldVel, cellStart, cellEnd,
								dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, &n_collision,k);
						else
							delta_x += ContactCell(neighbourPos, index, originalIndex, pos, vel,
							gridParticleIndex, Pos_exp, oldVel, cellStart, cellEnd,
							dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, &n_collision, k);*/


						if (IterType == Pre_Stablization)
							delta_x += MyContactCell(neighbourPos, index, originalIndex, pos, vel,
							gridParticleIndex, oldPos, oldPos, oldVel, dRelative,cellStart, cellEnd,
							dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, dObjectType, &n_collision);
						else
							delta_x += MyContactCell(neighbourPos, index, originalIndex, pos, vel,
							gridParticleIndex, oldPos, Pos_exp, oldVel, dRelative,cellStart, cellEnd,
							dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, dObjectType, &n_collision);
					}
				}
			}

			// collide with cursor sphere
			__syncthreads();
			delta_x += collideSpheres(pos_exp, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f)*delta_time;
			if (length(delta_x) > 0)
				int b = index;
			Pos_exp[originalIndex] += params.Omega * make_float4(delta_x, 0.0f) / 4;// max(1, n_collision);
			if (IterType == Pre_Stablization)
				oldPos[originalIndex] += params.Omega * make_float4(delta_x, 0.0f) / 4;// max(1, n_collision);
		}
	__global__
		void contactD_sorted(           // output: new velocity
		float4 *dPos,               // input: unsorted positions
		float4 *dPos_exp,
		float4 *SortedVel,            // input: unsorted velocities
		float4* SortedPos,
		float4* SortedPos_exp,
		float4* SortedRelative,
		float* SortedInvMss,
		int* SortedPhase,
		float* SortedSDF,
		float* SortedSDF_gradient,
		uint   *gridParticleIndex,    // input: sorted particle indices
		uint   *cellStart,
		uint   *cellEnd,	
		float* dRoate,
		ObjectType* dObjectType,
		uint    numParticles,
		float delta_time,
		IterationType IterType
		)
	{
			uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

			if (index >= numParticles) return;
			
			// read particle data from sorted arrays

			float3 pos;
			if (IterType == Pre_Stablization)
				pos = make_float3(FETCH(SortedPos, index));
			else
				pos = make_float3(FETCH(SortedPos_exp, index));
			float3 pos_exp = make_float3(FETCH(SortedPos_exp, index));
			float3 vel = make_float3(FETCH(SortedVel, index));

			// get address in grid
			int3 gridPos = calcGridPos(pos_exp);

			// examine neighbouring cells
			float3 delta_x = make_float3(0.0f);
			int n_collision = 0;
			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbourPos = gridPos + make_int3(x, y, z);



						if (IterType == Pre_Stablization)
							/*delta_x += MyContactCell(neighbourPos, index, originalIndex, pos, vel,
							gridParticleIndex, oldPos, oldPos, oldVel, dRelative, cellStart, cellEnd,
							dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, dObjectType, &n_collision);*/
							delta_x += MyContactCell_sort(neighbourPos, index, pos, gridParticleIndex, 
							SortedPos, SortedPos, SortedRelative, SortedInvMss, SortedSDF, SortedPhase, 
							(float4 *)SortedSDF_gradient, cellStart, cellEnd, dRoate, dObjectType, &n_collision);

						else
							/*delta_x += MyContactCell(neighbourPos, index, originalIndex, pos, vel,
							gridParticleIndex, oldPos, Pos_exp, oldVel, dRelative, cellStart, cellEnd,
							dInvMass, dSDF, dPhase, dSDF_gridiant, dRoate, dObjectType, &n_collision);*/
							delta_x += MyContactCell_sort(neighbourPos, index, pos_exp, gridParticleIndex,
							SortedPos, SortedPos_exp, SortedRelative, SortedInvMss, SortedSDF, SortedPhase, 
							(float4 *)SortedSDF_gradient, cellStart, cellEnd, dRoate, dObjectType, &n_collision);
					}
				}
			}

			// collide with cursor sphere
			__syncthreads();
			uint originalIndex = gridParticleIndex[index];
			delta_x += collideSpheres(pos_exp, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f)*delta_time*4;
			if (length(delta_x) > 0)
				int b = index;
			dPos_exp[originalIndex] += params.Omega * make_float4(delta_x, 0.0f)*params.Omega/ 4;// max(1, n_collision);
			SortedPos_exp[index] += params.Omega * make_float4(delta_x, 0.0f)*params.Omega / 4;
			if (IterType == Pre_Stablization)
			{
				dPos[originalIndex] += params.Omega * make_float4(delta_x, 0.0f)*params.Omega / 4;// max(1, n_collision);
				SortedPos[index] += params.Omega * make_float4(delta_x, 0.0f)*params.Omega / 4;
			}
				
		}
	template <
		int                     BLOCK_THREADS,
		int                     ITEMS_PER_THREAD,
		BlockReduceAlgorithm    ALGORITHM>
		__global__ void BlockSumKernel(
		float       *d_in,          // Tile of input
		float       *d_out)         // Tile aggregate
	{
			// Specialize BlockReduce type for our thread block
			typedef BlockReduce<float, BLOCK_THREADS, ALGORITHM> BlockReduceT;
			// Shared memory
			__shared__ typename BlockReduceT::TempStorage temp_storage;
			// Per-thread tile data
			float data[ITEMS_PER_THREAD];
			LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);
			clock_t start = clock();
			int aggregate = BlockReduceT(temp_storage).Sum(data);
			// Store aggregate and elapsed clocks
			if (threadIdx.x == 0)
			{
				*d_out = aggregate;
			}
		}
	template <
		int                     BLOCK_THREADS,
		BlockReduceAlgorithm    ALGORITHM>
		__global__ void Cal_A_Kernel(
			float* dCenter,
			float* dPos,
			float* dInvmass,
			uint* dObjectStart,
			uint* dObjectSize,
			uint* dObjectMass,
			float* dRelative
		)

	{
		// Specialize BlockReduce type for our thread block
		typedef BlockReduce<float, BLOCK_THREADS, ALGORITHM> BlockReduceT;
		// Shared memory
		__shared__ typename BlockReduceT::TempStorage temp_storage;
		int index_Object = blockIdx.x;

	}
	__global__ void Update_Relavite_kernel(float* dSp, float* dRelative, float* dA_qq, float* dA_qq_quad, float* new_A_qq, float* new_A_qq_quad, int num)
	{
		uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index < num)
		{
			float3 r = make_float3(
				dRelative[index * 4],
				dRelative[index * 4 + 1],
				dRelative[index * 4 + 2]
				);
			float Sp[3 * 3];
			memcpy(Sp, dSp, 3 * 3 * sizeof(float));
			r = MatrixMulVector(Sp, (float*)&r, 3);
			dRelative[index * 4] = r.x;
			dRelative[index * 4 + 1] = r.y;
			dRelative[index * 4 + 2] = r.z;
			float q_quad[9];
			q_quad[0] = r.x; q_quad[1] = r.y; q_quad[2] = r.z;
			q_quad[3] = r.x * r.x; q_quad[4] = r.y * r.y; q_quad[5] = r.z * r.z;
			q_quad[6] = r.x * r.y; q_quad[7] = r.y * r.z; q_quad[8] = r.z * r.x;
			MatrixMul((float*)&r, (float*)&r, new_A_qq + index * 3 * 3, 3, 1, 3);
			MatrixMul(q_quad, q_quad, new_A_qq_quad + index * 9 * 9, 9, 1, 9);
		}
	}
	/*template <int BLOCK_SIZE>
	__global__ void BlockReduction_Float(float* dData, float* dResult, int num)
	{
		__shared__ sTempMem[2 * BLOCK_SIZE];
		int num_remain = num;
		uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (2 * index < num)
			sTemp
	}*/
	template <int BLOCK_SIZE, int DATA_STEP>
	__global__ void Cal_Center_BR_kernel(float* dPos_exp, float* dCenter, float* dInvMass, uint* m_dObjectStart, uint* m_dObjectSize, ObjectType* dObjectType,float* m_dObjectMass)
	{
		__shared__ float sTempMem[BLOCK_SIZE * 2];
		//int DATA_STEP = 4;
		int index_object = blockIdx.x / DATA_STEP;
		int index_th = threadIdx.x;
		uint start = m_dObjectStart[index_object];
		uint size = m_dObjectSize[index_object];
		float mass = m_dObjectMass[index_object];
		ObjectType  type = dObjectType[index_object];
		///*if (type < Rigid || type > Deformation_quad_plasticity)
		//	return;*/
		float* Data = dPos_exp + start * DATA_STEP;
		float* Data_ivm = dInvMass + start;
		int offset = blockIdx.x % DATA_STEP;
		int num_remain = m_dObjectSize[index_object];
		if (index_th * DATA_STEP + offset < size * 4)
		{
			sTempMem[index_th] = Data[index_th * DATA_STEP + offset] / Data_ivm[index_th];
		}
		else
		{
			sTempMem[index_th] = 0;
		}
		if (index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset < size * 4)
		{
			sTempMem[index_th + BLOCK_SIZE] = Data[index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset] / Data_ivm[index_th + BLOCK_SIZE];
		}
		else
		{
			sTempMem[index_th + BLOCK_SIZE] = 0;
		}
		__syncthreads();
		num_remain -= 2 * BLOCK_SIZE;
		int i = 0;
		int index;
		while (num_remain > 0)
		{
			index = (2 * BLOCK_SIZE + index_th + i * BLOCK_SIZE) * DATA_STEP + offset;
			if (index < size * 4)
			{
				sTempMem[index_th] += Data[index] / Data_ivm[2 * BLOCK_SIZE + index_th + i * BLOCK_SIZE];
			}
			num_remain -= BLOCK_SIZE;
			i++;
		}
		__syncthreads();
		for (uint stride = BLOCK_SIZE; stride > 0; stride /= 2)
		{
			__syncthreads();
			if (index_th < stride)
			{
				sTempMem[index_th] += sTempMem[index_th + stride];
			}
		}
		__syncthreads();
		if (index_th == 0)
		{
			dCenter[index_object * 4 + offset] = sTempMem[0] / mass;
		}
	}

	template <int BLOCK_SIZE, int DATA_STEP>
	__global__ void Cal_Matrix_A_BR_kernel(float* A_input, float* A_result, uint* m_dObjectStart, uint* m_dObjectSize, ObjectType * dObjectType)
	{
		__shared__ float sTempMem[BLOCK_SIZE * 2];
		__shared__ ObjectType type;
		//int DATA_STEP = 4;
		int index_object = blockIdx.x / DATA_STEP;
		type = dObjectType[index_object];
		if (type < Rigid || type > Deformation_quad_plasticity)
			return;
		int index_th = threadIdx.x;
		uint start = m_dObjectStart[index_object];
		uint size = m_dObjectSize[index_object];
		float* Data = A_input + start * DATA_STEP;
		//float* Data_ivm = dInvMass + start;
		int offset = blockIdx.x % DATA_STEP;
		int num_remain = m_dObjectSize[index_object];
		if (index_th * DATA_STEP + offset < size * DATA_STEP)
		{
			sTempMem[index_th] = Data[index_th * DATA_STEP + offset];
		}
		else
		{
			sTempMem[index_th] = 0;
		}
		if (index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset < size * DATA_STEP)
		{
			sTempMem[index_th + BLOCK_SIZE] = Data[index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset];
		}
		else
		{
			sTempMem[index_th + BLOCK_SIZE] = 0;
		}
		__syncthreads();
		num_remain -= 2 * BLOCK_SIZE;
		int i = 0;
		int index;
		while (num_remain > 0)
		{
			index = (2 * BLOCK_SIZE + index_th + i * BLOCK_SIZE) * DATA_STEP + offset;
			if (index < size * DATA_STEP)
			{
				sTempMem[index_th] += Data[index];
			}
			num_remain -= BLOCK_SIZE;
			i++;
		}
		__syncthreads();
		for (uint stride = BLOCK_SIZE; stride > 0; stride /= 2)
		{
			__syncthreads();
			if (index_th < stride)
			{
				sTempMem[index_th] += sTempMem[index_th + stride];
			}
		}
		__syncthreads();
		if (index_th == 0)
		{
			A_result[index_object * 3 * 3 + offset] = sTempMem[0];
		}
	}
	template <int BLOCK_SIZE, int DATA_STEP>
	__global__ void Cal_Matrix_A_quad_BR_kernel(float* A_input, float* A_result, uint* m_dObjectStart, uint* m_dObjectSize, ObjectType * dObjectType)
	{
		__shared__ float sTempMem[BLOCK_SIZE * 2];
		__shared__ ObjectType type;
		//int DATA_STEP = 4;
		int index_object = blockIdx.x / DATA_STEP;
		type = dObjectType[index_object];
		if (type < Deformation_quad || type > Deformation_quad_plasticity)
			return;
		int index_th = threadIdx.x;
		uint start = m_dObjectStart[index_object];
		uint size = m_dObjectSize[index_object];
		float* Data = A_input + start * DATA_STEP;
		//float* Data_ivm = dInvMass + start;
		int offset = blockIdx.x % DATA_STEP;
		int num_remain = m_dObjectSize[index_object];
		if (index_th * DATA_STEP + offset < size * DATA_STEP)
		{
			sTempMem[index_th] = Data[index_th * DATA_STEP + offset];
		}
		else
		{
			sTempMem[index_th] = 0;
		}
		if (index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset < size * DATA_STEP)
		{
			sTempMem[index_th + BLOCK_SIZE] = Data[index_th * DATA_STEP + BLOCK_SIZE* DATA_STEP + offset];
		}
		else
		{
			sTempMem[index_th + BLOCK_SIZE] = 0;
		}
		__syncthreads();
		num_remain -= 2 * BLOCK_SIZE;
		int i = 0;
		int index;
		while (num_remain > 0)
		{
			index = (2 * BLOCK_SIZE + index_th + i * BLOCK_SIZE) * DATA_STEP + offset;
			if (index < size * DATA_STEP)
			{
				sTempMem[index_th] += Data[index];
			}
			num_remain -= BLOCK_SIZE;
			i++;
		}
		__syncthreads();
		for (uint stride = BLOCK_SIZE; stride > 0; stride /= 2)
		{
			__syncthreads();
			if (index_th < stride)
			{
				sTempMem[index_th] += sTempMem[index_th + stride];
			}
		}
		__syncthreads();
		if (index_th == 0)
		{
			A_result[index_object * DATA_STEP + offset] = sTempMem[0];
		}
	}
	__global__ void Boundary_kernel(float3 MaxB, float3 MinB, float4* oldPos, float4* dPos_exp, float r, int num_particle, IterationType itType)
	{
		uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		float3 x_delta = make_float3(0, 0, 0);
		if (index < num_particle)
		{
			float3 pos;
			float3 pos_exp;
			pos_exp = make_float3(
				dPos_exp[index].x,
				dPos_exp[index].y,
				dPos_exp[index].z);
			if (itType == Main_Constaint)
				pos = make_float3(
					dPos_exp[index].x,
					dPos_exp[index].y,
					dPos_exp[index].z);
			else
				pos = make_float3(
				oldPos[index].x,
				oldPos[index].y,
				oldPos[index].z);
			float d;
			for (int dim = 0; dim < 3; dim++)
			{

				if (((float*)&pos_exp)[dim] >((float*)&MaxB)[dim] - r)
				{
					float3 vect = make_float3(0, 0, 0);
					((float*)&vect)[dim] = 1;
					d = ((float*)&MaxB)[dim] - r - ((float*)&pos)[dim];
					
					x_delta += d * vect * 4 / params.Omega;
					if (params.friction)
					{
						float3 pos_old = make_float3(
							oldPos[index].x,
							oldPos[index].y,
							oldPos[index].z);
						float3 x_del = pos - pos_old;
						vect = make_float3(0, 0, 0);
						((float*)&vect)[dim] = 1;
						float3 x_del_tri = x_del - vect * ((float*)&x_del)[dim];
						if (length(x_del_tri) > 1e-5)
						{
							if (length(x_del_tri) < params.factor_s*abs(d))
							{
								x_del_tri = x_del_tri;
							}
							else
							{
								x_del_tri = x_del_tri * min_float(params.factor_k * abs(d) / length(x_del_tri), 1);
							}
						}
						x_delta += -x_del_tri;

					}
					
				}
				if (((float*)&pos_exp)[dim]< ((float*)&MinB)[dim] + r)
				{
					float3 vect = make_float3(0, 0, 0);
					((float*)&vect)[dim] = 1;
					d = ((float*)&MinB)[dim] + r - ((float*)&pos)[dim];
					x_delta += d * vect * 4 / params.Omega;
					if (params.friction)
					{
						float3 pos_old = make_float3(
							oldPos[index].x,
							oldPos[index].y,
							oldPos[index].z);
						float3 x_del = pos - pos_old;
						vect = make_float3(0, 0, 0);
						((float*)&vect)[dim] = 1;
						float3 x_del_tri = x_del - vect * ((float*)&x_del)[dim];
						if (length(x_del_tri) > 1e-5)
						{
							if (length(x_del_tri) < params.factor_s* abs(d))
							{
								x_del_tri = x_del_tri;
							}
							else
							{
								x_del_tri = x_del_tri * min_float(params.factor_k * abs(d) / length(x_del_tri), 1);
							}
						}
						


						x_delta += -x_del_tri;

					}
					
				}
			}
			dPos_exp[index].x += params.Omega * x_delta.x/4;
			dPos_exp[index].y += params.Omega * x_delta.y / 4;
			dPos_exp[index].z += params.Omega * x_delta.z / 4;
			if (itType == Pre_Stablization)
			{
				oldPos[index].x += params.Omega * x_delta.x / 4;
				oldPos[index].y += params.Omega * x_delta.y / 4;
				oldPos[index].z += params.Omega * x_delta.z / 4;
			}
		}
	}
#endif

	         