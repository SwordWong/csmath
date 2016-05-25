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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
//#include <nvMatrix.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
//#include <Eigen/Dense>  
//#include <Eigen/Eigenvalues>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvMatrix.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include <thrust/execution_policy.h>
#include "thrust/sort.h"
#include <thrust/device_vector.h>
#include "particles_kernel_impl.cuh"
#include "particleSystem.h"
#include <list>

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset,  long long int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float *pos,
                         float *vel,
                         float deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
            integrate_functor(deltaTime));
    }

	void my_integrateSystem(float *pos, 
		float *pos_exp,
		float *vel,
		float deltaTime,
		uint numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_pos_exp4((float4 *)pos_exp);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);

		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_pos_exp4, d_vel4)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_pos_exp4 + numParticles, d_vel4 + numParticles)),
			my_integrate_functor(deltaTime));
	}

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float4 *) sortedPos,
            (float4 *) sortedVel,
            gridParticleHash,
            gridParticleIndex,
            (float4 *) oldPos,
            (float4 *) oldVel,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
    }


	void reorderDataAndFindCellStart_sort(uint  *cellStart,
		uint  *cellEnd,
		float *sortedPos,
		float *sortedPos_exp,
		float *sortedVel,
		float *sortedRelative,
		float *sortedSDF,
		float *sortedSDFgradient,
		int *sortedPhase,
		float *sortedInvMass,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		float *oldPos,
		float *oldPos_exp,
		float *oldVel,
		float *oldRelative,
		float *oldSDF,
		float *oldSDFgradient,
		int *oldPhase,
		float *oldInvMass,
		uint   numParticles,
		uint   numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif

		uint smemSize = sizeof(uint)*(numThreads + 1);
		reorderDataAndFindCellStartD_sort << < numBlocks, numThreads, smemSize >> >(
			cellStart,
			cellEnd,
			(float4 *)sortedPos,
			(float4 *)sortedPos_exp,
			(float4 *)sortedVel,
			(float4 *)sortedRelative,
			sortedSDF,
			(float4 *)sortedSDFgradient,
			sortedPhase,
			sortedInvMass,
			gridParticleHash,
			gridParticleIndex,
			(float4 *)oldPos,
			(float4 *)oldPos_exp,
			(float4 *)oldVel,
			(float4 *)oldRelative,
			oldSDF,
			(float4 *)oldSDFgradient,
			oldPhase,
			oldInvMass,
			numParticles,
			numCells);
		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
	}



    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif 

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((float4 *)newVel,
                                              (float4 *)sortedPos,
                                              (float4 *)sortedVel,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
                                              numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }
	struct Center_functor
	{
		Center_functor(){};
		__host__ __device__
		float4 operator()(const float &invmass ,const float4 &pos ) const{
			return pos / invmass;
		}
	};
	struct Pos_Plus
	{
		Pos_Plus(){};
		__host__ __device__
			float4 operator()(const float4 &a, const float4 &b) const{
				return a+b;
			}
	};
	//void Cal_Center(float* dPos_exp, float* dInvMass,uint* ObjectStart, uint *ObjectSize, float* dObjecMass, float* dCenter)
	//{
	//	float4 *start;
	//	float4 center[MAX_OBJECT];
	//	int index = (int)ObjectStart;
	//	int size;
	//	float mass;
	//	
	//	//center = (float4*)&dPos_exp[index * 4];
	//	for (int i = 0; i < params.numObject; i++)
	//	{
	//		int index = (int)ObjectStart + i;
	//		size = ObjectSize[index];
	//		mass = dObjecMass[index];
	//		thrust::device_ptr<float4> d_pos_exp4((float4 *)dPos_exp);
	//		thrust::device_vector<float4> vec_dpos_exp4(d_pos_exp4, d_pos_exp4 + size);

	//		thrust::device_ptr<float> d_invmass((float *)dInvMass);
	//		thrust::device_vector<float> vec_invmass(d_invmass, d_invmass + size);

	//		thrust::transform(vec_invmass.begin(), vec_invmass.end(), vec_dpos_exp4.begin(), vec_dpos_exp4.end(), Center_functor());
	//		center[i] = thrust::reduce(vec_dpos_exp4.begin(),vec_dpos_exp4.end(),make_float4(0),thrust::plus<float4>());
	//		center[i] /= mass;
	//	}
	//}
	void Cal_Center_BR(float* dPos_exp, float* dCenter, float* dInvMass, uint* m_dObjectStart, uint* m_dObjectSize,ObjectType* dObjectType, float* m_dObjectMass , int num_object)
	{
		uint numThreads, numBlocks;
		numBlocks = num_object * 4;
		numThreads = 128;
		Cal_Center_BR_kernel<128,4> <<< numBlocks, numThreads >>>(dPos_exp, dCenter, dInvMass, m_dObjectStart, m_dObjectSize, dObjectType,m_dObjectMass);

	}
	float4 Cal_Center(float* dPos_exp, float* dInvMass, uint Start, uint Size, float dObjecMass)
	{
		float4 *start;
		float4 center;

		float *Fx, *Fy, *Fz, *Fw;
		int index = Start;
		//int size;
		//float mass;
		allocateArray((void**)&Fx, Size*sizeof(float));
		allocateArray((void**)&Fy, Size*sizeof(float));
		allocateArray((void**)&Fz, Size*sizeof(float));
		allocateArray((void**)&Fw, Size*sizeof(float));
		uint numThreads, numBlocks;
		computeGridSize(Size, 256, numBlocks, numThreads);
		
		thrust::device_ptr<float4> d_pos_exp4((float4 *)(&dPos_exp[4*index]));
		thrust::device_vector<float4> vec_dpos_exp4(d_pos_exp4, d_pos_exp4 + Size);
		thrust::device_ptr<float> d_invmass((float *)(dInvMass + index));
		thrust::device_vector<float> vec_invmass(d_invmass, d_invmass + Size);

		thrust::transform(vec_invmass.begin(), vec_invmass.end(), vec_dpos_exp4.begin(), vec_dpos_exp4.end(), Center_functor());

		Float4ToFloat <<<numBlocks, numThreads >>>(Size, (float4*)&dPos_exp[index*4], Fx, Fy, Fz, Fw);
		thrust::device_ptr<float> dFx((float *)Fx);
		thrust::device_ptr<float> dFy((float *)Fy);
		thrust::device_ptr<float> dFz((float *)Fz);
		thrust::device_ptr<float> dFw((float *)Fw);
		//center = thrust::reduce(vec_dpos_exp4.begin(), vec_dpos_exp4.end(), init, Pos_Plus());
		center.x = thrust::reduce(dFx, dFx + Size, 0.0f, thrust::plus<float>());
		center.y = thrust::reduce(dFy, dFy + Size, 0.0f, thrust::plus<float>());
		center.z = thrust::reduce(dFz, dFz + Size, 0.0f, thrust::plus<float>());
		//center.w = thrust::reduce(dFw, dFw + Size, 0.0f, thrust::plus<float>());
		center.w = 1;
		//printf("after reduce\n");
		freeArray(Fx);
		freeArray(Fy);
		freeArray(Fz);
		freeArray(Fw);
		center = center / dObjecMass;
		return center;
	}
	void Cal_Matrix_A(int size, float4 center, float* dPos_exp, float* dRelative, float* result)
	{
		uint numThreads, numBlocks;
		float* A_acc;
		float* temp;
		float tmp[400];

		allocateArray((void**)&A_acc, size * 3 * 3 * sizeof(float));
		allocateArray((void**)&temp, size*sizeof(float));
		
		computeGridSize(size, 256, numBlocks, numThreads);
		Cal_Matrix_A_element_kernel<<<numBlocks, numThreads >>>(size, center,dPos_exp,dRelative,A_acc);
		cudaMemcpy(tmp, A_acc, 400 * sizeof(float), cudaMemcpyDeviceToHost);
		//for (int i = )
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				MatrixToFloat <<<numBlocks, numThreads >>>(A_acc,temp,i,j,size,3,3);
				thrust::device_ptr<float> dtemp(temp);
				result[i * 3 + j] = thrust::reduce(dtemp, dtemp + size, 0.0f, thrust::plus<float>());
			}
		}
		freeArray(A_acc);
		freeArray(temp);
	}
	void Cal_Matrix_A_BR(float* dPos_exp, float* dCenter, int* dPhase, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType,float* result, int num_Particle, int num_Object)
	{
		uint numThreads, numBlocks;
		float* A_acc;
		float* A;
		//float tmp[4000];
		//float tmp2[4000];
		//float* temp;
		allocateArray((void**)&A_acc, num_Particle * 3 * 3 * sizeof(float));
		allocateArray((void**)&A, num_Object * 3 * 3 * sizeof(float));
		//computeGridSize(num_Particle, 256, numBlocks, numThreads);
		numBlocks = num_Object;
		numThreads = 256;
		Cal_Matrix_A_element_BR_kernal << <numBlocks, numThreads >> >(dPos_exp, dCenter, dObjectStart, dObjectSize, dRelative, dObjectType, A_acc, num_Particle);
		//cudaMemcpy(tmp, A_acc,3* 3* 5 * sizeof(float), cudaMemcpyDeviceToHost);

		//cudaMemcpy(tmp2, A_acc + 5 * 3 * 3, 5 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		//for (int index_obj = 0; index_obj < num_Object; index_obj++)
		//{
		//	int size = 125; //dObjectSize[index_obj];
		//	int start = index_obj * 125;// dObjectStart[index_obj];
		//	float4 center;
		//	cudaMemcpy((float*)&center, dCenter + 4 * index_obj, 4 * sizeof(float),cudaMemcpyDeviceToHost);
		//	Cal_Matrix_A_element_kernel << <numBlocks, numThreads >> >(size, center, dPos_exp + 4 * start, dRelative + 4 * start, A_acc + 3*3 * start);
		//	allocateArray((void**)&temp, size*sizeof(float));
		//	/*for (int i = 0; i < 3; i++)
		//	{
		//		for (int j = 0; j < 3; j++)
		//		{
		//			MatrixToFloat << <numBlocks, numThreads >> >(A_acc + start *  3 * 3, temp, i, j, size, 3, 3);
		//			thrust::device_ptr<float> dtemp(temp);
		//			result[index_obj  * 3 * 3 + i * 3 + j] = thrust::reduce(dtemp, dtemp + size, 0.0f, thrust::plus<float>());
		//		}
		//	}*/
		//	freeArray(temp);
		//}
		


		numBlocks = num_Object * 3 * 3;
		numThreads = 128;
		Cal_Matrix_A_BR_kernel<128, 3 * 3> << <numBlocks, numThreads >> >(A_acc, A, dObjectStart, dObjectSize, dObjectType);
		cudaMemcpy(result, A, 3 * 3 * num_Object*sizeof(float), cudaMemcpyDeviceToHost);
		freeArray(A);
		freeArray(A_acc);
	}
	void Cal_Matrix_A_quad(int size, float4 center, float* dPos_exp, float* dRelative, float* result)
	{
		uint numThreads, numBlocks;
		float* A_acc;
		float* temp;

		allocateArray((void**)&A_acc, size * 3 * 9 * sizeof(float));
		allocateArray((void**)&temp, size*sizeof(float));


		computeGridSize(size, 256, numBlocks, numThreads);
		Cal_Matrix_A_quad_element_kernel << <numBlocks, numThreads >> >(size, center, dPos_exp, dRelative, A_acc);

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				MatrixToFloat << <numBlocks, numThreads >> >(A_acc, temp, i, j, size,3,9);
				thrust::device_ptr<float> dtemp(temp);
				result[i * 9 + j] = thrust::reduce(dtemp, dtemp + size, 0.0f, thrust::plus<float>());
			}
		}
		freeArray(A_acc);
		freeArray(temp);
	}

	void Cal_Matrix_A_quad_BR(float* dPos_exp, float* dCenter, int* dPhase, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType, float* result, int num_Particle, int num_Object)
	{
		uint numThreads, numBlocks;
		float* A_acc;
		float* A_quad;
		//float tmp[4000];
		//float tmp2[4000];
		//float* temp;
		allocateArray((void**)&A_acc, num_Particle * 3 * 9 * sizeof(float));
		allocateArray((void**)&A_quad, num_Object * 3 * 9 * sizeof(float));

		numBlocks = num_Object;
		numThreads = 256;
		Cal_Matrix_A_quad_element_BR_kernel << <numBlocks, numThreads >> >(dPos_exp, dCenter, dObjectStart, dObjectSize, dRelative, dObjectType, A_acc, num_Particle);
		numBlocks = num_Object * 3 * 9;
		numThreads = 128;
		Cal_Matrix_A_quad_BR_kernel<128, 3 * 9> << <numBlocks, numThreads >> >(A_acc, A_quad, dObjectStart, dObjectSize, dObjectType);
		cudaMemcpy(result, A_quad, 3 * 9 * num_Object*sizeof(float), cudaMemcpyDeviceToHost);
		freeArray(A_quad);
		freeArray(A_acc);
	}


	void Shaping_Upadate_Pos_Exp(float *dPos_exp, float* dRelative, int *dPhase, float* dCenter, float *dRoate, float* dRoate_quad,ObjectType *dObjectType,uint num_Particles)
	{
		uint numThreads, numBlocks;
		computeGridSize(num_Particles, 256, numBlocks, numThreads);
		Shaping_Cal_pos_kernel << <numBlocks, numThreads >> >(dPos_exp, dRelative, dPhase, dCenter, dRoate, dRoate_quad,dObjectType, num_Particles);
	}
	//void Update_Mesh(float* dVertex, float* dNormal, uint* dFace, float *dRelative, float* dRoate, float* dRoate_quad, float* dcenter, int num_face, int num_vertex, ObjectType type)
	//{
	//	uint numThreads, numBlocks;
	//	computeGridSize(num_vertex, 256, numBlocks, numThreads);
	//	Update_Vertex << <numBlocks, numThreads >> >(dVertex, dNormal, dRelative, dRoate, dRoate_quad, dcenter, num_vertex,type);
	//	computeGridSize(MAX(num_vertex,num_face), 256, numBlocks, numThreads);
	//	int *num_tri_per_point;
	//	//num_tri_per_point = (int*)malloc(num_vertex*sizeof(int));
	//	/*if (type == Rigid)
	//		return;*/
	//	allocateArray((void **)&num_tri_per_point, num_vertex*sizeof(int));
	//	cudaMemset(num_tri_per_point, 0, num_vertex*sizeof(uint));
	//	cudaMemset(dNormal, 0, num_vertex * sizeof(float)* 3);
	//	//cudaMalloc(&num_tri_per_point, num_vertex*sizeof(int));
	//	Update_Normal << <numBlocks, numThreads >> >(dVertex, dFace, dNormal,num_face, num_vertex);
	//	cudaFree(num_tri_per_point);
	//	computeGridSize(num_vertex, 256, numBlocks, numThreads);
	//	Normalize_Normal << <numBlocks, numThreads >> >(dNormal, num_vertex);
	//
	//	
	//	//freeArray(num_tri_per_point);
	//}
	void Update_MeshFullPal(float* dVertex, float* dNormal, uint* dFace, float* Normal_Original, ObjectType *dObjectType, float *dRelative, int * dPhaseVertex, int * dPhaseFace, float* dRoate, float* dRoate_quad, float* dcenter, int num_face, int num_vertex)
	{
		uint numThreads, numBlocks;
		/*float t1[300],t2[300];
		float c[4 * 100];
		int f[300];
		cudaMemcpy(t1,dRoate, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(c, dcenter, 2 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(f, dFace, 3 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(t2, dRelative, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("c1 (%f, %f, %f)\nc2 (%f, %f, %f)\n", c[0], c[1], c[2], c[5], c[6], c[7]);
		for (int i = 0; i < 18; i++)
		{
			printf("roate[%d] = %f\n", i, t1[i]);
		}
		for (int i = 0; i < 10; i++)
		{
			printf("f[%d] (%d, %d, %d)\n", i,f[i * 3], f[i * 3 + 1], f[i * 3 + 2]);
		}
		for (int i = 0; i < 10; i++)
		{
			printf("r[%d] (%f, %f, %f)\n", i, t2[i * 3], t2[i * 3 + 1], t2[i * 3 + 2]);
		}*/
		//cudaMemcpy(t1, dNormal, 300 * sizeof(float), cudaMemcpyDeviceToHost);
		computeGridSize(num_vertex, 256, numBlocks, numThreads);
		//cudaMemcpy(t1, dVertex, 300 * sizeof(float), cudaMemcpyDeviceToHost);
		Update_VertexFullPal << <numBlocks, numThreads >> >(dVertex, dNormal, Normal_Original, dRelative, dPhaseVertex, dObjectType, dRoate, dRoate_quad, dcenter, num_vertex);
		//cudaMemcpy(t2, dVertex, 300 * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(t2, dNormal, 300 * sizeof(float), cudaMemcpyDeviceToHost);


		computeGridSize(MAX(num_face,num_vertex), 256, numBlocks, numThreads);
		


		/*for (int i = 0; i < 10; i++)
		{
			printf("i = %d, n1(%f, %f, %f), n2(%f, %f, %f)\n", i, t1[i * 3], t1[i * 3 + 1], t1[i * 3 + 2], t2[i * 3], t2[i * 3 + 1], t2[i * 3 + 2]);
		}*/

		
		///*for (int i = 0; i < 10; i++)
		//{
		//	printf("i = %d, v1(%f, %f, %f), v2(%f, %f, %f)\n", i, t1[i * 3], t1[i * 3 + 1], t1[i * 3 + 2], t2[i * 3], t2[i * 3 + 1], t2[i * 3 + 2]);
		//}*/
		//
		////int *num_tri_per_point;
		////num_tri_per_point = (int*)malloc(num_vertex*sizeof(int));
		///*if (type == Rigid)
		//return;*/
		////allocateArray((void **)&num_tri_per_point, num_vertex*sizeof(int));
		////cudaMemset(num_tri_per_point, 0, num_vertex*sizeof(uint));
		//cudaMemset(dNormal, 0, num_vertex * sizeof(float)* 3);
	
		//cudaMemcpy(t1, dNormal, 300 * sizeof(float), cudaMemcpyDeviceToHost);
		Update_Normal << <numBlocks, numThreads >> >(dVertex, dFace, dNormal, dPhaseFace, dPhaseVertex,dObjectType, num_face, num_vertex);
		//cudaMemcpy(t2, dNormal, 300 * sizeof(float), cudaMemcpyDeviceToHost);




		/*for (int i = 0; i < 10; i++)
		{
			printf("i = %d, n1(%f, %f, %f), n2(%f, %f, %f)\n", i, t1[i * 3], t1[i * 3 + 1], t1[i * 3 + 2], t2[i * 3], t2[i * 3 + 1], t2[i * 3 + 2]);
		}*/



		//cudaFree(num_tri_per_point);
		computeGridSize(num_vertex, 256, numBlocks, numThreads);
		Normalize_Normal << <numBlocks, numThreads >> >(dNormal, dPhaseVertex,dObjectType ,num_vertex);


		//freeArray(num_tri_per_point);
	}
	void Update_Change(float *dPos, float *dPos_exp, float *dVel, int numParticles, float delta_time,float epsilon)

	{
		uint numThreads, numBlocks;

		computeGridSize(numParticles, 1024, numBlocks, numThreads);
		
		Update_Change_kernel <<<numBlocks, numThreads >>>(dPos, dPos_exp,dVel, numParticles, delta_time,epsilon);
	}

	void contact(float *newVel,
		float *dPos,
		float *dPos_exp, 
		float *dVel,
		float *dRelative,
		uint  *gridParticleIndex,
		uint  *cellStart,
		uint  *cellEnd,
		float*	dInvMass,
		float* dSDF,
		int* dPhase,
		float4*	dSDF_gridiant,
		float* dRoate,
		ObjectType* dObjectType,
		uint   numParticles,
		uint   numCells,
		float delta_time,
		IterationType IterType
		)
	{
#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
		checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

		// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		
		contactD <<<numBlocks, numThreads >>>((float4 *)newVel,
			(float4 *)dPos,
			(float4 *)dPos_exp,
			(float4 *)dVel,
			(float4 *)dRelative,
			gridParticleIndex,
			cellStart,
			cellEnd,
			dInvMass,
			dSDF,
			dPhase,
			dSDF_gridiant,
			dRoate,
			dObjectType,
			numParticles,
			delta_time,
			IterType
			);
		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
		checkCudaErrors(cudaUnbindTexture(cellStartTex));
		checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
	}






	void contact_sort(
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
#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
		checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

		// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		contactD_sorted << <numBlocks, numThreads >> >(           // output: new velocity
			dPos,             // input: unsorted positions
			dPos_exp,
			SortedVel,            // input: unsorted velocities
			SortedPos,
			SortedPos_exp,
			SortedRelative,
			SortedInvMss,
			SortedPhase,
			SortedSDF,
			SortedSDF_gradient,
			gridParticleIndex,    // input: sorted particle indices
			cellStart,
			cellEnd,
			dRoate,
			dObjectType,
			numParticles,
			delta_time,
			IterType
			);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
		checkCudaErrors(cudaUnbindTexture(cellStartTex));
		checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
	}







	void Update_dRelative(float* dSp, float* dRelavite, float* A_qq, float* A_qq_quad, int num)
	{
		uint numThreads, numBlocks;
		computeGridSize(num, 64, numBlocks, numThreads);
		float* new_A_qq;
		float* new_A_qq_quad;
		allocateArray((void**)&new_A_qq, num * 3 * 3 * sizeof(float));
		allocateArray((void**)&new_A_qq_quad, num * 9 * 9 * sizeof(float));
		Update_Relavite_kernel <<<numBlocks, numThreads >>> (dSp, dRelavite, A_qq, A_qq_quad, new_A_qq, new_A_qq_quad, num);
		float* temp;
		allocateArray((void**)&temp, num * sizeof(float));
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				MatrixToFloat <<<numBlocks, numThreads >>>(new_A_qq, temp, i, j, num, 3, 3);
				thrust::device_ptr<float> dtemp(temp);
				float a = thrust::reduce(dtemp, dtemp + num, 0.0f, thrust::plus<float>());
				A_qq[i * 3 + j] = thrust::reduce(dtemp, dtemp + num, 0.0f, thrust::plus<float>());
			}
		}
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				MatrixToFloat <<<numBlocks, numThreads >>>(new_A_qq_quad, temp, i, j, num, 9, 9);
				thrust::device_ptr<float> dtemp(temp);
				A_qq_quad[i * 9 + j] = thrust::reduce(dtemp, dtemp + num, 0.0f, thrust::plus<float>());
			}
		}
	}
	void Boundary(float3 MaxB, float3 MinB, float4* oldPos, float4* dPos_exp, float r, int num_particle, IterationType itType)
	{
		uint numThreads, numBlocks;
		computeGridSize(num_particle, 64, numBlocks, numThreads);
		Boundary_kernel << <numBlocks, numThreads >> > (MaxB, MinB, oldPos, dPos_exp, r, num_particle, itType);
	}

}   // extern "C"
