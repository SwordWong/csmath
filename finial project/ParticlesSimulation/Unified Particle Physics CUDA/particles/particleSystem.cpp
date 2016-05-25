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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include "Object.h"
#include <Eigen/Dense>  
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <iostream>
#include <PlyLoader.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;











ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(0),
	m_numObjects(0),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;
	m_params.numObject = m_numObjects;

    m_params.particleRadius = 2.0f / 64.0f;
    m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.2f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	//m_params.gravity = make_float3(0.0f, -0.03f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

//inline float lerp(float a, float b, float t)
//{
//    return a + t*(b-a);
//}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::       _initialize(int numParticles)
{
    assert(!m_bInitialized);
	out.open("5.txt");
	if (!out)
		printf("fail to open file\n");
    m_numParticles = 0;
	m_numObjects = 0;
	m_numVertexes = 0;
	m_numFaces = 0;
	num_iter = 0;

    // allocate host storage
	m_hPos = new float[MAX_PARTICLES * 4];
	memset(m_hPos, 0, MAX_PARTICLES * 4 * sizeof(float));
	/*for (int i = 0; i < 10; i++)
	{
		printf("m_hPos[%d] = %d\n", i, m_hPos[i]);
	}*/



	m_hVel = new float[MAX_PARTICLES * 4];
	memset(m_hVel, 0, MAX_PARTICLES * 4 * sizeof(float));


	m_hInvm = new float[MAX_PARTICLES];
	memset(m_hInvm, 0, MAX_PARTICLES *  sizeof(float));


	m_hPhase = new int[MAX_PARTICLES];
	memset(m_hPhase, 0, MAX_PARTICLES *  sizeof(int));

	m_hSDF = new float[MAX_PARTICLES];
	memset(m_hSDF, 0, MAX_PARTICLES *  sizeof(float));

	m_hSDF_gradient = new float[MAX_PARTICLES * 4];
	memset(m_hSDF_gradient, 0, MAX_PARTICLES * 4 * sizeof(float));


	m_hRelative = new float[MAX_PARTICLES * 4];
	memset(m_hRelative, 0, MAX_PARTICLES * 4 * sizeof(float));
	
	m_hForce = new float[MAX_PARTICLES * 4];
	memset(m_hForce, 0, MAX_PARTICLES * 4 * sizeof(float));
	
	
	m_hPhaseVertex = new int[MAX_VERTEX];
	m_hPhaseFace = new int[MAX_FACE];


	m_hVertexStart = new int[MAX_OBJECT];
	m_hFaceStart = new int[MAX_OBJECT];

	m_hObjectNumVertex = new uint[MAX_OBJECT];
	m_hObjectNumFace = new uint[MAX_OBJECT];
	
	
	
    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	m_hObjectSize = new uint[MAX_OBJECT];
	m_hObjectStart = new uint[MAX_OBJECT];
	m_hObjectEnd = new uint[MAX_OBJECT];
	m_hObjectType = new ObjectType[MAX_OBJECT];
	m_hRoate = new float[MAX_OBJECT * 3 * 3];
	m_hRoate_quad = new float[MAX_OBJECT * 3 * 9];
	m_hObjectMass = new float[MAX_OBJECT];
	m_hA_qq = new float[MAX_OBJECT * 3 * 3];
	m_hA_qq_quad = new float[MAX_OBJECT * 9 * 9];
	m_hObjectSp = new float[MAX_OBJECT * 3 * 3];

	memset(m_hObjectSize, 0, MAX_OBJECT *  sizeof(uint));
	memset(m_hObjectEnd, 0, MAX_OBJECT *  sizeof(uint));
	memset(m_hObjectStart, 0, MAX_OBJECT *  sizeof(uint));
	memset(m_hObjectType, 0, MAX_OBJECT *  sizeof(ObjectType));
	memset(m_hRoate, 0, 3*3*MAX_OBJECT*sizeof(float));
	memset(m_hObjectMass, 0, MAX_OBJECT *  sizeof(float));
	memset(m_hObjectSp, 0, 3 * 3 * MAX_OBJECT*sizeof(float));
	
	//m_hPM = new ParticleMesh[MAX_OBJECT];
	
	/*for (int i = 0; i < 100; i++)
	{
		printf("m_hObjectSize[%d] = %d\n", i, m_hObjectSize[i]);
	}*/
    // allocate GPU data
	unsigned int memSize = sizeof(float)* 4 * MAX_PARTICLES;

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

		m_vertexVbo = createVBO(MAX_VERTEX * 3 * sizeof(float));
		registerGLBufferObject(m_vertexVbo, &m_cuda_vertexvbo_resource);

		m_indexVBO = createVBO(3 * MAX_FACE * sizeof(uint));
		registerGLBufferObject(m_indexVBO, &m_cuda_indexvbo_resource);

		m_vertexcolorVBO = createVBO(MAX_VERTEX * 3 * sizeof(float));
		registerGLBufferObject(m_vertexcolorVBO, &m_cuda_vertexcolorvbo_resource);

		m_NorVBO = createVBO(MAX_VERTEX * 3 * sizeof(float));
		registerGLBufferObject(m_NorVBO, &m_cuda_normalvbo_resource);

		m_NorOriginalVBO = createVBO(MAX_VERTEX * 3 * sizeof(float));
		registerGLBufferObject(m_NorVBO, &m_cuda_normal_original_vbo_resource);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    }

    allocateArray((void **)&m_dVel, memSize);
	allocateArray((void **)&m_dPos_exp, memSize);
	//allocateArray((void**)&m_dPos_contact,memSize);
	allocateArray((void **)&m_dInvm, sizeof(float)*MAX_PARTICLES);
	allocateArray((void **)&m_dPhase, sizeof(float)*MAX_PARTICLES);
	allocateArray((void **)&m_dSDF, sizeof(float)*MAX_PARTICLES);
	allocateArray((void **)&m_dSDF_gradient, memSize);
	allocateArray((void **)&m_dRelative, memSize);
	allocateArray((void **)&m_dForce, memSize);
	//allocateArray((void **)&m_dRoate, memSize*4);

	allocateArray((void **)&m_dRelativeVertex, sizeof(float)*3*MAX_VERTEX);

	allocateArray((void **)&m_dPhaseVertex, sizeof(int)*MAX_VERTEX);
	allocateArray((void **)&m_dPhaseFace, sizeof(int)*MAX_FACE);

	allocateArray((void **)&m_dVertexStart, sizeof(int)*MAX_OBJECT);
	allocateArray((void **)&m_dFaceStart, sizeof(int)*MAX_OBJECT);

	allocateArray((void **)&m_dObjectNumVertex, sizeof(int)*MAX_OBJECT);
	allocateArray((void **)&m_dObjectNumFace, sizeof(int)*MAX_OBJECT);












    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dSortedPos_exp, memSize);
	allocateArray((void **)&m_dSortedRelative, memSize);
	allocateArray((void **)&m_dSortedSDF_gradient, memSize);
	allocateArray((void **)&m_dSortedSDF, MAX_PARTICLES * sizeof(float));
	allocateArray((void **)&m_dSortedPhase, MAX_PARTICLES * sizeof(int));
	allocateArray((void **)&m_dSortedInvMass, MAX_PARTICLES * sizeof(float));
	allocateArray((void **)&m_dSortedObectType, MAX_PARTICLES * sizeof(ObjectType));

	allocateArray((void **)&m_dGridParticleHash, MAX_PARTICLES*sizeof(uint));
	allocateArray((void **)&m_dGridParticleIndex, MAX_PARTICLES*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));
	
	allocateArray((void **)&m_dObjectSize, MAX_OBJECT*sizeof(uint));
	allocateArray((void **)&m_dObjectStart, MAX_OBJECT*sizeof(uint));
	allocateArray((void **)&m_dObjectEnd, MAX_OBJECT*sizeof(uint)); 
	allocateArray((void **)&m_dObjectType, MAX_OBJECT*sizeof(ObjectType));
	allocateArray((void **)&m_dRoate, MAX_OBJECT*3*3*sizeof(float));
	allocateArray((void **)&m_dObjectMass, MAX_OBJECT * sizeof(float));
	allocateArray((void **)&m_dCenter, 4 * MAX_OBJECT * sizeof(float));
	allocateArray((void **)&m_dA_qq, MAX_OBJECT * 3 * 3 * sizeof(float));
	allocateArray((void **)&m_dSp, MAX_OBJECT * 3 * 3 * sizeof(float));
	allocateArray((void **)&m_dA_qq_quad, MAX_OBJECT * 9 * 9 * sizeof(float));
	allocateArray((void **)&m_dRoate_quad, MAX_OBJECT * 3 * 9 * sizeof(float));
    if (m_bUseOpenGL)
    {
		m_colorVBO = createVBO(MAX_PARTICLES * 4 * sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i=0; i<MAX_PARTICLES; i++)
        {
            float t = i / (float) MAX_PARTICLES;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
			//printf("i = %d\n", i);
            colorRamp(t, ptr);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
	delete [] m_hInvm;
	delete[] m_hPhase;
	delete[] m_hSDF;
	delete[] m_hForce;
	delete[] m_hSDF_gradient;
	delete[] m_hRelative;
	delete[] m_hObjectSize;
	delete[] m_hObjectStart;
	delete[] m_hObjectEnd;
	delete[] m_hObjectType;
	delete[] m_hRoate;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;
	delete[] m_dCenter;
	delete[] m_hA_qq;
	delete[] m_hA_qq_quad;
	delete[] m_hPhaseFace;
	delete[] m_hPhaseVertex;
	delete[] m_hVertexStart;
	delete[] m_hFaceStart;
	delete[] m_hObjectNumFace;
	delete[] m_hObjectNumVertex;




    freeArray(m_dVel);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
	freeArray(m_dInvm);
	freeArray(m_dPhase);
	freeArray(m_dSDF);
	freeArray(m_dSDF_gradient);	
	freeArray(m_dRelative);
	freeArray(m_dForce);
	freeArray(m_dRoate);
	freeArray(m_dA_qq);
	freeArray(m_dA_qq_quad);
	//freeArray(m_dPos_contact);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

	freeArray(m_dObjectStart);
	freeArray(m_dObjectEnd);
	freeArray(m_dObjectSize);
	freeArray(m_dObjectType);
	freeArray(m_dObjectMass);

	freeArray(m_dPhaseFace);
	freeArray(m_dPhaseVertex);
	freeArray(m_dVertexStart);
	freeArray(m_dFaceStart);
	freeArray(m_dObjectNumFace);
	freeArray(m_dObjectNumVertex);
	freeArray(m_dRelativeVertex);


    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);

		unregisterGLBufferObject(m_cuda_indexvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_indexVBO);

		unregisterGLBufferObject(m_cuda_normalvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_NorVBO);

		unregisterGLBufferObject(m_cuda_vertexcolorvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_vertexcolorVBO);

		unregisterGLBufferObject(m_cuda_vertexvbo_resource);
		glDeleteBuffers(1, (const GLuint *)&m_vertexVbo);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

// step the simulation
void 
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;
	int k = 5;
	float factor_s = 5;
	float factor_k = 2.1;
	bool friction = true;
    if (m_bUseOpenGL)
    {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
    }
    else
    {
        dPos = (float *) m_cudaPosVBO;
    }
	//printf("iteration:%d\n", num_iter);
    // update constants
    setParameters(&m_params);

    // integrate
	
	my_integrateSystem(dPos, m_dPos_exp, m_dVel, deltaTime, m_numParticles);

	
	/*shaping();
	Update_Change(dPos, m_dPos_exp, m_dPos_contact, m_dVel, m_numParticles, deltaTime, m_params.particleRadius / 100);*/
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		m_dPos_exp,
		m_numParticles);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
	/*uint t_dGridParticleHash[10000], t_dGridParticleIndex[10000];
	memset(t_dGridParticleHash, -1, 10000);
	cudaMemcpy((float*)t_dGridParticleHash, m_dGridParticleHash, 100  * sizeof(uint), cudaMemcpyDeviceToHost);
	memset(t_dGridParticleIndex, -1, 10000);
	cudaMemcpy((float*)t_dGridParticleIndex, m_dGridParticleIndex, 100 * sizeof(uint), cudaMemcpyDeviceToHost);*/
	//for (int i = 0; i < 54; i++)
	//{
	//	//printf("iter = %d, particle %d, (%8f, %8f, %8f, %8f)\n",num_iter,i,t[i].x,t[i].y,t[i].z,t[i].w);
	//	cout << "t_dGridParticleHash: iter = " << num_iter << ",particle_" << i << " " << t_dGridParticleHash[i] << endl;
	//	cout << "t_dGridParticleIndex: iter = " << num_iter << ",particle_" << i << " " << t_dGridParticleIndex[i] << endl;
	//}
	// reorder particle arrays into sorted order and
	// find start and end of each cell
	/*reorderDataAndFindCellStart(
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		m_dPos_exp,
		m_dVel,
		m_numParticles,
		m_numGridCells);*/
	reorderDataAndFindCellStart_sort(
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedPos_exp,
		m_dSortedVel,
		m_dSortedRelative,
		m_dSortedSDF,
		m_dSortedSDF_gradient,
		m_dSortedPhase,
		m_dSortedInvMass,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
		m_dPos_exp,
		m_dVel,
		m_dRelative,
		m_dSDF,
		m_dSDF_gradient,
		m_dPhase,
		m_dInvm,
		m_numParticles,
		m_numGridCells
		);
	float3 MaxB = make_float3(1, 99, 1);
	float3 MinB = make_float3(-1, -1, -1);
	/*float3 MaxB = make_float3(2, 99, 2);
	float3 MinB = make_float3(-2, -1, -2);*/
	//pre-stabilization
	for (int i = 0; i <1; i++)
	{
		// process collisions
		/*collide(
			m_dVel,
			dPos,
			m_dVel,
			m_dGridParticleIndex,
			m_dCellStart,
			m_dCellEnd,
			m_numParticles,
			m_numGridCells);*/
		Boundary(MaxB, MinB, (float4*)dPos, (float4*)m_dPos_exp, m_params.particleRadius, m_numParticles, Pre_Stablization);
		/*contact(m_dVel, dPos, m_dPos_exp, m_dVel, m_dRelative,m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
			m_dInvm, m_dSDF, m_dPhase, (float4*)m_dSDF_gradient, m_dRoate, m_dObjectType, m_numParticles, m_numGridCells, deltaTime,Pre_Stablization);*/
		
		contact_sort((float4*)dPos, (float4 *)m_dPos_exp, (float4*)m_dSortedVel, (float4*)m_dSortedPos, (float4*)m_dSortedPos_exp,
			(float4 *)m_dSortedRelative, m_dSortedInvMass, m_dSortedPhase, m_dSortedSDF, m_dSortedSDF_gradient, m_dGridParticleIndex, m_dCellStart,
			m_dCellEnd, m_dRoate, m_dObjectType, m_numParticles, deltaTime, Pre_Stablization);
	}
	//main constaint
	for (int i = 0; i < 2; i++)
	{
		
		Boundary(MaxB, MinB, (float4*)dPos, (float4*)m_dPos_exp, m_params.particleRadius, m_numParticles, Main_Constaint);

		/*contact(m_dVel, dPos, m_dPos_exp, m_dVel, m_dRelative,m_dGridParticleIndex, m_dCellStart, m_dCellEnd,
			m_dInvm, m_dSDF, m_dPhase, (float4*)m_dSDF_gradient, m_dRoate, m_dObjectType, m_numParticles, m_numGridCells, deltaTime, Main_Constaint);*/

		contact_sort((float4*)dPos, (float4 *)m_dPos_exp, (float4*)m_dSortedVel, (float4*)m_dSortedPos, (float4*)m_dSortedPos_exp,
			(float4 *)m_dSortedRelative, m_dSortedInvMass, m_dSortedPhase, m_dSortedSDF, m_dSortedSDF_gradient, m_dGridParticleIndex, m_dCellStart,
			m_dCellEnd, m_dRoate, m_dObjectType, m_numParticles, deltaTime, Main_Constaint);

		//shaping(deltaTime);
		shaping_BR(deltaTime);
	}
	//shaping(deltaTime);
	//memset(t_exp, -1, 10000);
	//cudaMemcpy((float*)t_exp, m_dPos_exp, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//memset(t_pos, -1, 10000);
	//cudaMemcpy((float*)t_pos, dPos, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//memset(t_contact, -1, 10000);
	////cudaMemcpy((float*)t_contact, m_dPos_contact, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << "************************************************************************" << endl;
	//for (int i = 27; i < 54; i++)
	//{
	//	//printf("iter = %d, particle %d, (%8f, %8f, %8f, %8f)\n",num_iter,i,t[i].x,t[i].y,t[i].z,t[i].w);
	//	cout << "pos: iter = " << num_iter << ",particle_" << i << " (" << t_pos[i].x << ", " << t_pos[i].y << ", " << t_pos[i].z << ", " << t_pos[i].w << ")" << endl;
	//	cout << "exp: iter = " << num_iter << ",particle_" << i << " (" << t_exp[i].x << ", " << t_exp[i].y << ", " << t_exp[i].z << ", " << t_exp[i].w << ")" << endl;
	//	//cout << "contact: iter = " << num_iter << ",particle_" << i << " (" << t_contact[i].x << ", " << t_exp[i].y << ", " << t_exp[i].z << ", " << t_exp[i].w << ")" << endl;
	//}
	
	Update_Change(dPos, m_dPos_exp,  m_dVel, m_numParticles, deltaTime, m_params.particleRadius / 50);
	/*float4 centers[MAX_OBJECT];
	for (int i = 0; i < m_numObjects; i++)
	{
		centers[i] = Cal_Center(m_dPos_exp, m_dInvm, m_hObjectStart[i], m_hObjectSize[i], m_hObjectMass[i]);
	}
	setArray(OBJECT_CENTER, centers, 0, m_numObjects);*/
	if (m_params.mesh && m_numVertexes > 0)
	{
		//for (int i = 0; i < m_numObjects; i++)
		//{
		//	/*float t1[300];
		//	float t2[300];
		//	float c[4 * 100];*/
		//	if (m_hPM[i].hasMesh)
		//	{
		//		ObjectType type = m_hObjectType[i];
		//		float* dVertex = (float *)mapGLBufferObject(&m_hPM[i].m_cuda_vertexvbo_resource);
		//		float* dNormal = (float *)mapGLBufferObject(&m_hPM[i].m_cuda_normalvbo_resource);
		//		uint* dFace = (uint *)mapGLBufferObject(&m_hPM[i].m_cuda_indexvbo_resource);
		//		/*cudaMemcpy(t1, dVertex, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		//		cudaMemcpy(c, m_dCenter, 2 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		//		printf("c1 (%f, %f, %f)\nc2 (%f, %f, %f)\n", c[0], c[1], c[2], c[5], c[6], c[7]);*/
		//		Update_Mesh(dVertex, dNormal, dFace, (float*)m_hPM[i].dRelative_Vertex, m_dRoate + i * 3 * 3, NULL, m_dCenter + i * 4, m_hPM[i].m_totalFaces, m_hPM[i].m_totalConnectedPoints, type);
		//		//cudaMemcpy(t2, dVertex, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		//		unmapGLBufferObject(m_hPM[i].m_cuda_vertexvbo_resource);
		//		//printf("vertex unmap\n");
		//		unmapGLBufferObject(m_hPM[i].m_cuda_normalvbo_resource);
		//		//printf("normal unmap\n");
		//		unmapGLBufferObject(m_hPM[i].m_cuda_indexvbo_resource);
		//		//printf("face unmap\n");
		//	}
		//}


		/*float t1[300];
		float t2[300];
		float c[4 * 100];*/
		float* dVertex = (float *)mapGLBufferObject(&m_cuda_vertexvbo_resource);
		float* dNormal = (float *)mapGLBufferObject(&m_cuda_normalvbo_resource);
		float* dNormal_Original = (float*)mapGLBufferObject(&m_cuda_normal_original_vbo_resource);
		uint* dFace = (uint *)mapGLBufferObject(&m_cuda_indexvbo_resource);
		/*cudaMemcpy(t1, m_dRoate, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(c, m_dCenter, 2 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		printf("c1 (%f, %f, %f)\nc2 (%f, %f, %f)\n", c[0], c[1], c[2], c[5], c[6], c[7]);
		for (int i = 0; i < 18; i++)
		{
			printf("roate[i] = %f\n", t1[i]);
		}*/
		Update_MeshFullPal(dVertex, dNormal, dFace, dNormal_Original, m_dObjectType, m_dRelativeVertex, m_dPhaseVertex, m_dPhaseFace,m_dRoate, m_dRoate_quad, m_dCenter, m_numFaces, m_numVertexes);
		//cudaMemcpy(t2, dVertex, 100 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		unmapGLBufferObject(m_cuda_vertexvbo_resource);
		unmapGLBufferObject(m_cuda_normalvbo_resource);
		unmapGLBufferObject(m_cuda_indexvbo_resource);
		unmapGLBufferObject(m_cuda_normal_original_vbo_resource);

	}
	



	//cout << "after update" << endl;
	//memset(t_exp, -1, 10000);
	//cudaMemcpy((float*)t_exp, m_dPos_exp, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//memset(t_pos, -1, 10000);
	//cudaMemcpy((float*)t_pos, dPos, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//memset(t_contact, -1, 10000);
	//cudaMemcpy((float*)t_contact, m_dPos_contact, 54 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 27; i < 54; i++)
	//{
	//	//printf("iter = %d, particle %d, (%8f, %8f, %8f, %8f)\n",num_iter,i,t[i].x,t[i].y,t[i].z,t[i].w);
	//	cout << "pos: iter = " << num_iter << ",particle_" << i << " (" << t_pos[i].x << ", " << t_pos[i].y << ", " << t_pos[i].z << ", " << t_pos[i].w << ")" << endl;
	//	cout << "exp: iter = " << num_iter << ",particle_" << i << " (" << t_exp[i].x << ", " << t_exp[i].y << ", " << t_exp[i].z << ", " << t_exp[i].w << ")" << endl;
	//	cout << "contact: iter = " << num_iter << ",particle_" << i << " (" << t_contact[i].x << ", " << t_exp[i].y << ", " << t_exp[i].z << ", " << t_exp[i].w << ")" << endl;
	//}
	//cout << "************************************************************************" << endl;
//     note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
	num_iter++;
	
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

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
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
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const void *data, int start, int count)
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
		case VERTEX:
		{
					   unregisterGLBufferObject(m_cuda_vertexvbo_resource);
					   glBindBuffer(GL_ARRAY_BUFFER, m_vertexVbo);
					   glBufferSubData(GL_ARRAY_BUFFER, start * 3 * sizeof(float), count * 3 * sizeof(float), data);
					   glBindBuffer(GL_ARRAY_BUFFER, 0);
					   registerGLBufferObject(m_vertexVbo, &m_cuda_vertexvbo_resource);
					   break;
		}
		case FACE:
		{
					   unregisterGLBufferObject(m_cuda_indexvbo_resource);
					   glBindBuffer(GL_ARRAY_BUFFER, m_indexVBO);
					   glBufferSubData(GL_ARRAY_BUFFER, start * 3 * sizeof(int), count * 3 * sizeof(int), data);
					   glBindBuffer(GL_ARRAY_BUFFER, 0);
					   registerGLBufferObject(m_indexVBO, &m_cuda_indexvbo_resource);
					   break;
		}
		case VERTEX_COLOR:
		{
					unregisterGLBufferObject(m_cuda_vertexcolorvbo_resource);
					   glBindBuffer(GL_ARRAY_BUFFER, m_vertexcolorVBO);
					   glBufferSubData(GL_ARRAY_BUFFER, start * 3 * sizeof(float), count * 3 * sizeof(float), data);
					   glBindBuffer(GL_ARRAY_BUFFER, 0);
					   registerGLBufferObject(m_vertexcolorVBO, &m_cuda_vertexcolorvbo_resource);
					   break;
		}
		case NORMAL:
		{
							 unregisterGLBufferObject(m_cuda_normalvbo_resource);
							 glBindBuffer(GL_ARRAY_BUFFER, m_NorVBO);
							 glBufferSubData(GL_ARRAY_BUFFER, start * 3 * sizeof(float), count * 3 * sizeof(float), data);
							 glBindBuffer(GL_ARRAY_BUFFER, 0);
							 registerGLBufferObject(m_NorVBO, &m_cuda_normalvbo_resource);
							 break;
		}
		case NORMAL_ORIGINAL:
		{
					   unregisterGLBufferObject(m_cuda_normal_original_vbo_resource);
					   glBindBuffer(GL_ARRAY_BUFFER, m_NorOriginalVBO);
					   glBufferSubData(GL_ARRAY_BUFFER, start * 3 * sizeof(float), count * 3 * sizeof(float), data);
					   glBindBuffer(GL_ARRAY_BUFFER, 0);
					   registerGLBufferObject(m_NorOriginalVBO, &m_cuda_normal_original_vbo_resource);
					   break;
		}
		case OBJECT_VERTEX_START:
			copyArrayToDevice(m_dVertexStart, data, start * sizeof(int), count * sizeof(int));
			break;
		case OBJECT_FACE_START:
			copyArrayToDevice(m_dFaceStart, data, start * sizeof(int), count * sizeof(int));
			break;
		case PHASE_VERTEX:
			copyArrayToDevice(m_dPhaseVertex, data, start * sizeof(int), count * sizeof(int));
			break;
		case PHASE_FACE:
			copyArrayToDevice(m_dPhaseFace, data, start * sizeof(int), count * sizeof(int));
			break;
		case OBJECT_NUM_VERTEX:
			copyArrayToDevice(m_dObjectNumVertex, data, start * sizeof(uint), count * sizeof(uint));
			break;
		case OBJECT_NUM_FACE:
			copyArrayToDevice(m_dObjectNumFace, data, start * sizeof(uint), count * sizeof(uint));
			break;
		case RELATIVE_VERTEX:
			copyArrayToDevice(m_dRelativeVertex, data, start * 3 * sizeof(float), count * 3 * sizeof(float));
			break;
        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
		case INVERSE_MASS:
			copyArrayToDevice(m_dInvm, data, start*sizeof(float), count*sizeof(float));
			break;
		case PHASE:
			copyArrayToDevice(m_dPhase, data, start*sizeof(int), count*sizeof(int));
			break;
		case SDF_arr:
			copyArrayToDevice(m_dSDF, data, start*sizeof(float), count*sizeof(float));
			break;
		case SDF_GRADIENT:
			copyArrayToDevice(m_dSDF_gradient, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
			break;
		case RELATIVE_POS:
			copyArrayToDevice(m_dRelative, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
			break;
		case FORCE:
			copyArrayToDevice(m_dForce, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
			break;
		case ROATE:
			copyArrayToDevice(m_dRoate, data, start * 3 * 3 * sizeof(float), count * 3 * 3 * sizeof(float));
			break;
		case ROATE_QUAD:
			copyArrayToDevice(m_dRoate_quad, data, start * 3 * 9 * sizeof(float), count * 3 * 9 * sizeof(float));
			break;
		case OBJECT_SIZE:
			copyArrayToDevice(m_dObjectSize, data, start*sizeof(uint), count*sizeof(uint));
			break;
		case OBJECT_START:
			copyArrayToDevice(m_dObjectStart, data, start*sizeof(uint), count*sizeof(uint));
			break;
		case OBJECT_END:
			copyArrayToDevice(m_dObjectEnd, data, start*sizeof(uint), count*sizeof(uint));
			break;
		case OBJECT_TYPE:
			copyArrayToDevice(m_dObjectType, data, start*sizeof(uint), count*sizeof(uint));
			break;
		case OBJECT_MASS:
			copyArrayToDevice(m_dObjectMass, data, start*sizeof(float), count*sizeof(float));
			break;
		case  OBJECT_CENTER:
			copyArrayToDevice(m_dCenter, data, start*4*sizeof(float), count* 4 * sizeof(float));
			break;
		case OBJECT_A_QQ:
			copyArrayToDevice(m_dA_qq, data, start * 3 * 3 * sizeof(float), count * 3 * 3 * sizeof(float));
			break;
		case OBJECT_A_QQ_QUAD:
			copyArrayToDevice(m_dA_qq_quad, data, start * 9 * 9 * sizeof(float), count * 9 * 9 * sizeof(float));
			break;
		case OBJECT_SP:
			copyArrayToDevice(m_dSp, data, start * 3 * 3 * sizeof(float), count * 3 * 3 * sizeof(float));
			break;
		/*case POSITION_CONTACT:
			copyArrayToDevice(m_dPos_contact, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
			break;*/
		case EMPTY:break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = 1.0f;

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
	m_numParticles = 0;
	m_numObjects = 0;
	triple par_list[10000];
	triple start;
	//int size = 6;
	SDF sdf_list[10000];
	switch (config)
	{
	default:
	case CONFIG_RANDOM:
	{
						  int p = 0, v = 0;

						  for (uint i = 0; i < m_numParticles; i++)
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
						  setArray(POSITION, m_hPos, 0, m_numParticles);
						  setArray(VELOCITY, m_hVel, 0, m_numParticles);
						  break;
	}


	case CONFIG_GRID:
	{
						m_numParticles = 16384;
						float jitter = m_params.particleRadius*0.01f;
						uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
						uint gridSize[3];
						gridSize[0] = gridSize[1] = gridSize[2] = s;
						initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
						setArray(POSITION, m_hPos, 0, m_numParticles);
						setArray(VELOCITY, m_hVel, 0, m_numParticles);
	}
		break;
	case ADD_RIGID:
	{
					  /* for (int i = 0; i < 100; i++)
					   {
					   printf("m_hObjectSize[%d] = %d\n", i, m_hObjectSize[i]);
					   }*/
					  /*triple par_list[10000];
					  triple start;
					  SDF sdf_list[10000];*/
					  MakeCube(10, 10, 10, m_params.particleRadius, par_list, sdf_list);
					  /* for (int i = 0; i < 100; i++)
					   {
					   printf("m_hObjectSize[%d] = %d\n", i, m_hObjectSize[i]);
					   }*/
					  start.x[0] = start.x[1] = start.x[2] = 0;
					  //start.x[1] = -1;
					  addRigid(start, par_list, sdf_list, 10 * 10 * 10, 1, m_params.particleRadius);
					  /*float vel[4] = { 1, 0.0, 0.0 ,0.0};
					  float VEL[4 * 10000];
					  float *hv;
					  setArray(VELOCITY, vel, 0, 1);
					  hv = getArray(VELOCITY);
					  memcpy(VEL, hv, 1000 * sizeof(float));
					  int a = 1;*/
					  break;

	}
	case COLLISION_DEMO:
	{

						   m_numParticles = 0;
						   m_numObjects = 0;
						   m_numVertexes = 0;
						   m_numFaces = 0;
						
						   m_params.gridSize = m_gridSize;
						   m_params.numCells = m_numGridCells;
						   m_params.numBodies = m_numParticles;
						   m_params.numObject = m_numObjects;

						   m_params.particleRadius = 1.5f / 64.0f;
						   m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
						   m_params.colliderRadius = 0.2f;

						   m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
						   // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
						   float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
						   m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

						   //triple par_list[10000];
						   //triple start;
						   int size = 10;
						   //SDF sdf_list[10000];





						   LevelsetCollider LC;
						   CPLYLoader mesh;
						   mesh.LoadModel("cube.ply");
						   LC.setMeshModel(mesh);
						   LC.calculate(20, 0, 0);

						   ParticleMesh PM1, PM2;



						   //MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
						   PM1.calculate(LC, size, m_params.particleRadius, 1);

						   start.x[0] = 0;
						   start.x[1] = -1 + m_params.particleRadius;
						   start.x[2] = 0;
						   //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
						   addRigid(start, PM1.par_list, PM1.sdf_list, PM1.m_totalParticles, 1, m_params.particleRadius);
						   //addMesh(m_numObjects - 1, PM1);
						   addMesh(m_numObjects - 1, PM1, make_float3(1,1,0));

						   PM2.calculate(LC, size, m_params.particleRadius, 1);
						   start.x[0] = 0;
						   start.x[1] = 1.5 - 2 * size * m_params.particleRadius;
						   start.x[2] = 0;
						   //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
						   addRigid(start, PM2.par_list, PM2.sdf_list, PM2.m_totalParticles, 1, m_params.particleRadius);
						   //addMesh(m_numObjects - 1, PM2);
						   addMesh(m_numObjects - 1, PM2, make_float3(0, 1, 1));
						   break;

	}
	case STACK_DEMO:
	{
					   m_numParticles = 0;
					   m_numObjects = 0;
					   m_numVertexes = 0;
					   m_numFaces = 0;

					   m_params.gridSize = m_gridSize;
					   m_params.numCells = m_numGridCells;
					   m_params.numBodies = m_numParticles;
					   m_params.numObject = m_numObjects;

					   m_params.particleRadius = 1.5f / 64.0f;
					   m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
					   m_params.colliderRadius = 0.2f;

					   m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
					   // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
					   float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
					   m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

					   
					   LevelsetCollider LC;
					   CPLYLoader mesh;
					   mesh.LoadModel("cube.ply");
					   LC.setMeshModel(mesh);
					   LC.calculate(20, 0, 0);
					   int size = 6;
					   //SDF sdf_list[10000];
					   //MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
					   for (int i = 0; i < 5; i++)
					   {
						   ParticleMesh PM;
						   PM.calculate(LC, size, m_params.particleRadius, 1);
						   start.x[0] = -size*m_params.particleRadius;
						   start.x[1] = -1 + (i*size * 2 * 1)*m_params.particleRadius;
						   start.x[2] = -size*m_params.particleRadius;
						   //addRigid(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius);
						   addRigid(start, PM.par_list, PM.sdf_list, PM.m_totalParticles, 1, m_params.particleRadius);
						   //addDeformation(start, PM.par_list, PM.sdf_list, PM.m_totalParticles, 1, m_params.particleRadius, Deformation_linear);
						   addMesh(m_numObjects - 1, PM);
					   }


					   break;
	}
	case GRANULAR_FRICTION:
	{
							  m_numParticles = 0;
							  m_numObjects = 0;
							  m_numVertexes = 0;
							  m_numFaces = 0;

							  m_params.gridSize = m_gridSize;
							  m_params.numCells = m_numGridCells;
							  m_params.numBodies = m_numParticles;
							  m_params.numObject = m_numObjects;

							  m_params.particleRadius = 1.0f / 64.0f;
							  m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
							  m_params.colliderRadius = 0.2f;

							  m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
							  // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
							  float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
							  m_params.cellSize = make_float3(cellSize, cellSize, cellSize);


							  //triple par_list[10000];
							 // triple start;
							  int size = 20;
							  //SDF sdf_list[10000];
							  MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
							  for (int i = 0; i < 2; i++)
							  {
								  start.x[0] = -size*m_params.particleRadius;
								  start.x[1] = -1 + (i*size * 2 * 1 + 1)*m_params.particleRadius;
								  start.x[2] = -size*m_params.particleRadius;
								  addParticleGroup(start, par_list, size * size * size, 1, m_params.particleRadius);
							  }
							  break;
	}
	case DEFORMATION_LINEAR:
	{
							   m_numParticles = 0;
							   m_numObjects = 0;
							   m_numVertexes = 0;
							   m_numFaces = 0;


							   m_params.gridSize = m_gridSize;
							   m_params.numCells = m_numGridCells;
							   m_params.numBodies = m_numParticles;
							   m_params.numObject = m_numObjects;

							   m_params.particleRadius = 1.5f / 64.0f;
							   m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
							   m_params.colliderRadius = 0.2f;

							   m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
							   // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
							   float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
							   m_params.cellSize = make_float3(cellSize, cellSize, cellSize);


							   LevelsetCollider LC;
							   CPLYLoader mesh;
							   mesh.LoadModel("cube.ply");
							   LC.setMeshModel(mesh);
							   LC.calculate(20, 0, 0);

							   ParticleMesh PM1, PM2;
							  
							   int size = 10;
							 
							   //MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
							   PM1.calculate(LC, size, m_params.particleRadius, 1);
							   start.x[0] = 0;
							   start.x[1] = -1 + m_params.particleRadius;
							   start.x[2] = 0;
							   addDeformation(start, PM1.par_list, PM1.sdf_list, PM1.m_totalParticles, 1, m_params.particleRadius, Deformation_linear);
							   addMesh(m_numObjects - 1, PM1);

							   PM2.calculate(LC, size, m_params.particleRadius, 1);
							   start.x[0] = 0;
							   start.x[1] = 1.3 - 2 * size * m_params.particleRadius;
							   start.x[2] = size *  m_params.particleRadius;
							   addDeformation(start, PM2.par_list, PM2.sdf_list, PM2.m_totalParticles, 1, m_params.particleRadius, Deformation_linear);
							   addMesh(m_numObjects - 1, PM2);
							   break;

	}
	case DEFORMATION_QUAD:
	{
							 m_numParticles = 0;
							 m_numObjects = 0;
							 m_numVertexes = 0;
							 m_numFaces = 0;


							 m_params.gridSize = m_gridSize;
							 m_params.numCells = m_numGridCells;
							 m_params.numBodies = m_numParticles;
							 m_params.numObject = m_numObjects;

							 m_params.particleRadius = 1.5f / 64.0f;
							 m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
							 m_params.colliderRadius = 0.2f;

							 m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
							 // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
							 float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
							 m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
							
							 int size = 10;
							 
							 LevelsetCollider LC;
							 CPLYLoader mesh;
							 mesh.LoadModel("cube.ply");
							 LC.setMeshModel(mesh);
							 LC.calculate(20, 0, 0);

							 ParticleMesh PM1, PM2;
							 
							 //MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
							 PM1.calculate(LC, size, m_params.particleRadius, 1);
							 start.x[0] = 0;
							 start.x[1] = -1 + m_params.particleRadius;
							 start.x[2] = 0;
							 addDeformation(start, PM1.par_list, PM1.sdf_list, PM1.m_totalParticles, 1, m_params.particleRadius, Deformation_quad);
							 addMesh(m_numObjects - 1, PM1);

							 PM2.calculate(LC, size, m_params.particleRadius, 1);
							 start.x[0] = 0;
							 start.x[1] = 1.9 - 2 * size * m_params.particleRadius;
							 start.x[2] = size *  m_params.particleRadius;
							 addDeformation(start, PM2.par_list, PM2.sdf_list, PM2.m_totalParticles, 1, m_params.particleRadius, Deformation_quad);
							 addMesh(m_numObjects - 1, PM2);
							 break;

	}
	case DEFORMATION_LINEAR_PLA:
	{
								   m_numParticles = 0;
								   m_numObjects = 0;
								   m_numVertexes = 0;
								   m_numFaces = 0;


								   m_params.gridSize = m_gridSize;
								   m_params.numCells = m_numGridCells;
								   m_params.numBodies = m_numParticles;
								   m_params.numObject = m_numObjects;

								   m_params.particleRadius = 2.0f / 64.0f;
								   m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
								   m_params.colliderRadius = 0.2f;

								   m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
								   // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
								   float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
								   m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

								   //triple par_list[10000];
								   //triple start;
								   int size = 8;
								  // SDF sdf_list[10000];
								   MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
								   /* for (int i = 0; i < 100; i++)
								   {
								   printf("m_hObjectSize[%d] = %d\n", i, m_hObjectSize[i]);
								   }*/
								   start.x[0] = 0;
								   start.x[1] = -1 + m_params.particleRadius;
								   start.x[2] = 0;
								   //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
								   addDeformation(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius, Deformation_linear_plasticity);
								   start.x[0] = 0;
								   start.x[1] = 1 - 2 * size * m_params.particleRadius;
								   start.x[2] = size *  m_params.particleRadius;
								   //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
								   //addRigid(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius);
								   addDeformation(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius, Deformation_linear_plasticity);
								   break;
	}

	case DEFORMATION_QUAD_PLA:
	{
								 m_numParticles = 0;
								 m_numObjects = 0;
								 m_numVertexes = 0;
								 m_numFaces = 0;


								 m_params.gridSize = m_gridSize;
								 m_params.numCells = m_numGridCells;
								 m_params.numBodies = m_numParticles;
								 m_params.numObject = m_numObjects;

								 m_params.particleRadius = 2.0f / 64.0f;
								 m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
								 m_params.colliderRadius = 0.2f;

								 m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
								 // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
								 float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
								 m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

								 //triple par_list[10000];
								 //triple start;
								 int size = 6;
								 //SDF sdf_list[10000];
								 MakeCube(size, size, size, m_params.particleRadius, par_list, sdf_list);
								 /* for (int i = 0; i < 100; i++)
								 {
								 printf("m_hObjectSize[%d] = %d\n", i, m_hObjectSize[i]);
								 }*/
								 start.x[0] = 0;
								 start.x[1] = -1 + m_params.particleRadius;
								 start.x[2] = 0;
								 //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
								 addDeformation(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius, Deformation_quad_plasticity);
								 start.x[0] =  size *  m_params.particleRadius;
								 start.x[1] = 1 - 2 * size * m_params.particleRadius;
								 start.x[2] = size *  m_params.particleRadius;
								 //addRigid(start, par_list, sdf_list, 3 * 3 * 3, 1, m_params.particleRadius);
								 //addRigid(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius);
								 addDeformation(start, par_list, sdf_list, size * size * size, 1, m_params.particleRadius, Deformation_quad_plasticity);
								 break;

	}
	case MESH_DEMO:
	{
					  ParticleMesh PM1, PM2;//, PM3, PM4, PM5;
					  m_numParticles = 0;
					  m_numObjects = 0;
					  m_numVertexes = 0;
					  m_numFaces = 0;

							m_params.gridSize = m_gridSize;
							m_params.numCells = m_numGridCells;
							m_params.numBodies = m_numParticles;
							m_params.numObject = m_numObjects;

							//m_params.particleRadius = 1.0f / 64.0f/2;
							m_params.particleRadius = 1.0f / 64.0f;
							m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
							m_params.colliderRadius = 0.2f;

							m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
							// m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
							float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
							//float cellSize = m_params.particleRadius * 4.0f;  // cell size equal to particle diameter
							m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

							int size = 15;
							
							LevelsetCollider LC,LC1,LC2,LC3,LC4;
					
							CPLYLoader mesh;
							mesh.LoadModel("bunny.ply");
							LC.setMeshModel(mesh);
							LC.calculate(30,0,0);
							//m_hPM[m_numObjects].calculate(LC, size, m_params.particleRadius, 1);


							PM1.calculate(LC, size, m_params.particleRadius, 1);
							start.x[0] = 0;
							start.x[1] = -1;// +m_params.particleRadius * (1);
							start.x[2] = 0;
							addRigid(start, PM1.par_list, PM1.sdf_list, PM1.m_totalParticles, 1, m_params.particleRadius);
							addMesh(m_numObjects - 1, PM1);
							//PM1.release();

							start.x[0] = 0;
							start.x[1] = 1 -  2*size * m_params.particleRadius;
							start.x[2] = 0;
							PM2.calculate(LC, size, m_params.particleRadius, 1);
							addRigid(start, PM2.par_list, PM2.sdf_list, PM2.m_totalParticles, 1, m_params.particleRadius);
							addMesh(m_numObjects - 1, PM2);
						

							
							break;

	}
	case BUNNY_PILE:
	{
					   
					   m_numParticles = 0;
					   m_numObjects = 0;
					   m_numVertexes = 0;
					   m_numFaces = 0;

					  m_params.gridSize = m_gridSize;
					  m_params.numCells = m_numGridCells;
					  m_params.numBodies = m_numParticles;
					  m_params.numObject = m_numObjects;

					  m_params.particleRadius = 1.0f / 64.0f;
					  m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
					  m_params.colliderRadius = 0.2f;

					  m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
					  // m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
					  float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
					  m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

					  //triple par_list[20000];
					  //triple start;
					  int size = 11;
					  //SDF sdf_list[20000];
					  LevelsetCollider LC;
					  //ParticleMesh PM;
					  //int num = LoadFromPly("ball.ply", size, m_params.particleRadius, par_list, sdf_list);
					  CPLYLoader mesh;
					  mesh.LoadModel("bunny_simplify.ply");
					  LC.setMeshModel(mesh);
					  LC.calculate(20, 0, 0);
				


					   for (int i = 0; i < 15; i++)
					  {
						   ParticleMesh PM1, PM2, PM3, PM4;
						   start.x[0] = -(size + 3)*m_params.particleRadius;
						   start.x[1] = -1 + (i*size * 3 * 1 + 1)*m_params.particleRadius;
						   start.x[2] = -(size + 3)*m_params.particleRadius;
						   PM1.calculate(LC, size, m_params.particleRadius, 1);
						   addRigid(start, PM1.par_list, PM1.sdf_list, PM1.m_totalParticles, 1, m_params.particleRadius);
						   addMesh(m_numObjects - 1, PM1);
						   //PM1.release();

						   start.x[0] = (size + 3)*m_params.particleRadius;
						   start.x[1] = -1 + (i*size * 3 * 1 + 1)*m_params.particleRadius ;
						   start.x[2] = -(size + 3)*m_params.particleRadius;
						   PM2.calculate(LC, size, m_params.particleRadius, 1);
						   addRigid(start, PM2.par_list, PM2.sdf_list, PM2.m_totalParticles, 1, m_params.particleRadius);
						   addMesh(m_numObjects - 1, PM2);
						   //PM.release();

						   start.x[0] = -(size + 3)*m_params.particleRadius;
						   start.x[1] = -1 + (i*size * 3 * 1 + 1)*m_params.particleRadius ;
						   start.x[2] = (size + 3)*m_params.particleRadius;
						   PM3.calculate(LC, size, m_params.particleRadius, 1);
						   addRigid(start, PM3.par_list, PM3.sdf_list, PM3.m_totalParticles, 1, m_params.particleRadius);
						   addMesh(m_numObjects - 1, PM3);
						   //PM.release();

						   start.x[0] = (size + 3)*m_params.particleRadius;
						   start.x[1] = -1 + (i*size * 3 * 1 + 1)*m_params.particleRadius ;
						   start.x[2] = (size + 3)*m_params.particleRadius;
						   PM4.calculate(LC, size, m_params.particleRadius, 1);
						   addRigid(start, PM4.par_list, PM4.sdf_list, PM4.m_totalParticles, 1, m_params.particleRadius);
						   addMesh(m_numObjects - 1, PM4);
						   //PM.release();
					  }

					  break;

	}
	case NO_OPERATION: break;

	}
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

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

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
void 
ParticleSystem::addRigid(triple postion_start, triple  par_list[], SDF sdf_list[], int n_par_in_rig, float m, float r)
{
	triple postion_center;
	float x_acc[4] = { 0, 0, 0,1}; //accumulated weighted postion	by Zihao Wong
	float m_acc = 0;	//The mass of object
	float center[4] = { 0, 0, 0,0}; //weight center of object
	m_numObjects++;
	m_params.numObject = m_numObjects;
	m_hObjectSize[m_numObjects - 1] = (uint)n_par_in_rig;
	m_hObjectStart[m_numObjects - 1] = (uint)m_numParticles;
	m_hObjectEnd[m_numObjects - 1] = (uint)m_numParticles + n_par_in_rig;
	m_hObjectType[m_numObjects - 1] = Rigid;
	
	for (int i = 0; i < n_par_in_rig; i++)
	{
		int index = m_hObjectStart[m_numObjects - 1] + i;
		m_hPhase[index] = m_numObjects - 1;
		m_hInvm[index] = 1 / m;
		m_hSDF[index] = sdf_list[i].sdf;
		for (int j = 0; j < 3; j++)
		{
			m_hPos[index * 4 + j] = postion_start.x[j] + par_list[i].x[j];
			m_hVel[index * 4 + j] = 0;
			m_hForce[index * 4 + j] = 0;
			m_hSDF_gradient[index * 4 + j] = sdf_list[i].gradient[j];
			x_acc[j] += m_hPos[index * 4 + j] * m;
		}
		m_hPos[index * 4 + 3] = 1.0f;
		m_hVel[index * 4 + 3] = 0;
		m_hForce[index * 4 + 3] = 0;
		m_hSDF_gradient[index * 4 + 3] = 0;
		x_acc[3] += m_hPos[index * 4 + 3] * m;
		m_acc += m;
		m_numParticles++;
	}
	float T[500];
	//memcpy(T, m_hPos, m_numParticles*4 * sizeof(float));
	m_hObjectMass[m_numObjects - 1] = m_acc;
	//calculate the weight center
	for (int j = 0; j < 4; j++)
	{
		center[j] = x_acc[j] / m_acc;
	}
	//calculate particle's raleative position form weight center 
	for (int i = 0; i < n_par_in_rig; i++)
	{
		int index = m_hObjectStart[m_numObjects - 1] + i;
		for (int j = 0; j < 4; j++)
		{
			m_hRelative[index * 4 + j] = m_hPos[index * 4 + j] - center[j];
		}
	}
	//float a[1000];
	/*for (int i = 0; i < 1000; i++)\
		a[i] = m_hRelative[i];*/
	//set the roate matrix to identity
	for (int i = 0; i < 4; i++)
	{
		int index = i*4 + i;
		m_hRoate[(m_numObjects - 1) * 4 * 4 + index] = 1;
	}
	/*for (int i = 0; i < 1000; i++)\
		a[i] = m_hPos[i];*/
	/*for (int i = 0; i <100; i++)
	{
		printf("m_hPos[%d] = %f\n", i, m_hPos[i]);
	}*/
	setArray(POSITION, m_hPos + 4 * m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(VELOCITY, m_hVel + 4 * m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(INVERSE_MASS, m_hInvm + m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(PHASE, m_hPhase + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(SDF_arr, m_hSDF + m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(SDF_GRADIENT, m_hSDF_gradient + 4 * m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(RELATIVE_POS, m_hRelative + 4 * m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(FORCE, m_hForce + 4 * m_hObjectStart[m_numObjects - 1] , m_numParticles - n_par_in_rig, n_par_in_rig);
	setArray(ROATE, m_hRoate + 3*3 * (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_SIZE, m_hObjectSize, m_numObjects - 1, 1);
	setArray(OBJECT_START, m_hObjectStart + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_END, m_hObjectEnd + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_TYPE, m_hObjectType + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_MASS, m_hObjectMass + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_A_QQ, m_hA_qq + 3 * 3 * (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_A_QQ_QUAD, m_hA_qq_quad + 9 * 9 * (m_numObjects - 1), m_numObjects - 1, 1);
}
void ParticleSystem::addMesh(int index_obj, ParticleMesh &PM)
{
	int* Face = new int[PM.m_totalFaces * 3];
	m_hVertexStart[index_obj] = m_numVertexes;
	m_hFaceStart[index_obj] = m_numFaces;
	m_hObjectNumVertex[index_obj] = PM.m_totalConnectedPoints;
	m_hObjectNumFace[index_obj] = PM.m_totalFaces;
	for (int i = 0; i < PM.m_totalConnectedPoints; i++)
	{
		m_hPhaseVertex[m_hVertexStart[index_obj] + i] = index_obj;
	}
	for (int i = 0; i < PM.m_totalFaces; i++)
	{
		m_hPhaseFace[m_hObjectNumFace[index_obj] + i] = index_obj;
		Face[3 * i] = PM.FaceVerInd[3 * i] + m_numVertexes;
		Face[3 * i + 1] = PM.FaceVerInd[3 * i + 1] + m_numVertexes;
		Face[3 * i + 2] = PM.FaceVerInd[3 * i + 2] + m_numVertexes;
	}
	setArray(VERTEX, PM.mp_vertexXYZ, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(FACE, Face, m_numFaces, PM.m_totalFaces);
	setArray(NORMAL, PM.mp_vertexNorm, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(NORMAL_ORIGINAL, PM.mp_vertexNorm, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(VERTEX_COLOR, PM.mp_vertexRGB, m_numVertexes, PM.m_totalConnectedPoints);

	setArray(PHASE_VERTEX, m_hPhaseVertex + m_numVertexes, m_numVertexes, PM.m_totalConnectedPoints);

	setArray(PHASE_FACE, m_hPhaseFace + m_numFaces, m_numFaces, PM.m_totalFaces);
	setArray(OBJECT_VERTEX_START, m_hVertexStart + m_numObjects,  m_numObjects, 1);
	setArray(OBJECT_FACE_START, m_hFaceStart + m_numObjects, m_numObjects, 1);
	setArray(OBJECT_NUM_VERTEX, m_hObjectNumVertex + m_numObjects, m_numObjects, 1);
	setArray(OBJECT_NUM_FACE, m_hObjectNumFace + m_numObjects, m_numObjects, 1);
	//setArray(RELATIVE_VERTEX, (float*)&PM.Relative_Vertex, m_numVertexes, PM.m_totalConnectedPoints);
	cudaMemcpy(m_dRelativeVertex + 3 * m_numVertexes, PM.dRelative_Vertex, PM.m_totalConnectedPoints * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
	/*float t[300];
	cudaMemcpy(t, m_dRelativeVertex, 3 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		printf("m_dRelativeVertex[%d] = %f\n", i , t[i]);
	}*/
	m_numVertexes += PM.m_totalConnectedPoints;
	m_numFaces += PM.m_totalFaces;


}
void ParticleSystem::addMesh(int index_obj, ParticleMesh &PM, float3 color)
{
	int* Face = new int[PM.m_totalFaces * 3];
	float3* Color = new float3[PM.m_totalConnectedPoints];
	m_hVertexStart[index_obj] = m_numVertexes;
	m_hFaceStart[index_obj] = m_numFaces;
	m_hObjectNumVertex[index_obj] = PM.m_totalConnectedPoints;
	m_hObjectNumFace[index_obj] = PM.m_totalFaces;
	for (int i = 0; i < PM.m_totalConnectedPoints; i++)
	{
		m_hPhaseVertex[m_hVertexStart[index_obj] + i] = index_obj;
		Color[i] = color;
	}
	for (int i = 0; i < PM.m_totalFaces; i++)
	{
		m_hPhaseFace[m_hObjectNumFace[index_obj] + i] = index_obj;
		Face[3 * i] = PM.FaceVerInd[3 * i] + m_numVertexes;
		Face[3 * i + 1] = PM.FaceVerInd[3 * i + 1] + m_numVertexes;
		Face[3 * i + 2] = PM.FaceVerInd[3 * i + 2] + m_numVertexes;
	}
	setArray(VERTEX, PM.mp_vertexXYZ, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(FACE, Face, m_numFaces, PM.m_totalFaces);
	setArray(NORMAL, PM.mp_vertexNorm, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(NORMAL_ORIGINAL, PM.mp_vertexNorm, m_numVertexes, PM.m_totalConnectedPoints);
	setArray(VERTEX_COLOR, Color, m_numVertexes, PM.m_totalConnectedPoints);

	setArray(PHASE_VERTEX, m_hPhaseVertex + m_numVertexes, m_numVertexes, PM.m_totalConnectedPoints);

	setArray(PHASE_FACE, m_hPhaseFace + m_numFaces, m_numFaces, PM.m_totalFaces);
	setArray(OBJECT_VERTEX_START, m_hVertexStart + m_numObjects, m_numObjects, 1);
	setArray(OBJECT_FACE_START, m_hFaceStart + m_numObjects, m_numObjects, 1);
	setArray(OBJECT_NUM_VERTEX, m_hObjectNumVertex + m_numObjects, m_numObjects, 1);
	setArray(OBJECT_NUM_FACE, m_hObjectNumFace + m_numObjects, m_numObjects, 1);
	//setArray(RELATIVE_VERTEX, (float*)&PM.Relative_Vertex, m_numVertexes, PM.m_totalConnectedPoints);
	cudaMemcpy(m_dRelativeVertex + 3 * m_numVertexes, PM.dRelative_Vertex, PM.m_totalConnectedPoints * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
	/*float t[300];
	cudaMemcpy(t, m_dRelativeVertex, 3 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
	printf("m_dRelativeVertex[%d] = %f\n", i , t[i]);
	}*/
	m_numVertexes += PM.m_totalConnectedPoints;
	m_numFaces += PM.m_totalFaces;
	delete Face;
	delete Color;
}

void ParticleSystem::addDeformation(triple postion_start, triple  par_list[], SDF sdf_list[], int n_par_in_def, float m, float r, ObjectType type)
{
	triple postion_center;
	float x_acc[4] = { 0, 0, 0, 1 }; //accumulated weighted postion	by Zihao Wong
	float m_acc = 0;	//The mass of object
	float center[4] = { 0, 0, 0, 0 }; //weight center of object
	MatrixXf A_qq = Matrix3f::Zero(3, 3);
	MatrixXf A_qq_quad = MatrixXf::Zero(9, 9);
	m_numObjects++;
	m_params.numObject = m_numObjects;
	m_hObjectSize[m_numObjects - 1] = (uint)n_par_in_def;
	m_hObjectStart[m_numObjects - 1] = (uint)m_numParticles;
	m_hObjectEnd[m_numObjects - 1] = (uint)m_numParticles + n_par_in_def;
	if (type >= Deformation_linear && type <= Deformation_quad_plasticity)
		m_hObjectType[m_numObjects - 1] = type;
	else
	{
		printf("Invaild Deformation type\n");
		return;
	}


	for (int i = 0; i < n_par_in_def; i++)
	{
		int index = m_hObjectStart[m_numObjects - 1] + i;
		m_hPhase[index] = m_numObjects - 1;
		m_hInvm[index] = 1 / m;
		m_hSDF[index] = sdf_list[i].sdf;
		for (int j = 0; j < 3; j++)
		{
			m_hPos[index * 4 + j] = postion_start.x[j] + par_list[i].x[j];
			m_hVel[index * 4 + j] = 0;
			m_hForce[index * 4 + j] = 0;
			m_hSDF_gradient[index * 4 + j] = sdf_list[i].gradient[j];
			x_acc[j] += m_hPos[index * 4 + j] * m;
		}
		m_hPos[index * 4 + 3] = 1.0f;
		m_hVel[index * 4 + 3] = 0;
		m_hForce[index * 4 + 3] = 0;
		m_hSDF_gradient[index * 4 + 3] = 0;
		x_acc[3] += m_hPos[index * 4 + 3] * m;
		m_acc += m;
		m_numParticles++;
	}
	//memcpy(T, m_hPos, m_numParticles*4 * sizeof(float));
	m_hObjectMass[m_numObjects - 1] = m_acc;
	//calculate the weight center
	for (int j = 0; j < 4; j++)
	{
		center[j] = x_acc[j] / m_acc;
	}
	//calculate particle's raleative position form weight center 
	VectorXf qr(3, 1), q_quad(9, 1);
	for (int i = 0; i < n_par_in_def; i++)
	{
		int index = m_hObjectStart[m_numObjects - 1] + i;
		for (int j = 0; j < 4; j++)
		{
			m_hRelative[index * 4 + j] = m_hPos[index * 4 + j] - center[j];
		}
		float x = m_hRelative[index * 4];
		float y = m_hRelative[index * 4 + 1];
		float z = m_hRelative[index * 4 + 2];
		qr << x, y, z;
		q_quad << x, y, z, x*x, y*y, z*z, x*y, y*z, z*x;
		A_qq += qr * qr.transpose();
		A_qq_quad += q_quad * q_quad.transpose();
	}
	A_qq = A_qq.inverse();
	A_qq_quad = A_qq_quad.inverse();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int index = i * 3 + j;
			m_hA_qq[(m_numObjects - 1) * 3 * 3 + index] = A_qq(i, j);
		}
	}
	
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			int index = i * 9 + j;
			m_hA_qq_quad[(m_numObjects - 1) * 9 * 9 + index] = A_qq_quad(i, j);
		}
	}

	for (int i = 0; i < 3; i++)
	{
		int index = i * 3 + i;
		m_hRoate[(m_numObjects - 1) * 3 * 3 + index] = 1;
		m_hObjectSp[(m_numObjects - 1) * 3 * 3+ index] = 1;
	}
	


	setArray(POSITION, m_hPos + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(VELOCITY, m_hVel + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(INVERSE_MASS, m_hInvm + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(PHASE, m_hPhase + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(SDF_arr, m_hSDF + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(SDF_GRADIENT, m_hSDF_gradient + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(RELATIVE_POS, m_hRelative + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(FORCE, m_hForce + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_def, n_par_in_def);
	setArray(ROATE, m_hRoate + 3 * 3 * (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_SIZE, m_hObjectSize, m_numObjects - 1, 1);
	setArray(OBJECT_START, m_hObjectStart + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_END, m_hObjectEnd + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_TYPE, m_hObjectType + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_MASS, m_hObjectMass + (m_numObjects - 1), m_numObjects - 1, 1);
}
void ParticleSystem::addParticleGroup(triple postion_start, triple  par_list[], int n_par_in_group, float m, float r)
{
	
	m_numObjects++;
	m_params.numObject = m_numObjects;
	m_hObjectSize[m_numObjects - 1] = (uint)n_par_in_group;
	m_hObjectStart[m_numObjects - 1] = (uint)m_numParticles;
	m_hObjectEnd[m_numObjects - 1] = (uint)m_numParticles + n_par_in_group;
	m_hObjectType[m_numObjects - 1] = Particle;

	for (int i = 0; i < n_par_in_group; i++)
	{
		int index = m_hObjectStart[m_numObjects - 1] + i;
		m_hPhase[index] = m_numObjects - 1;
		m_hInvm[index] = 1 / m;
		//m_hSDF[index] = sdf_list[i].sdf;
		for (int j = 0; j < 3; j++)
		{
			m_hPos[index * 4 + j] = postion_start.x[j] + par_list[i].x[j];
			m_hVel[index * 4 + j] = 0;
			m_hForce[index * 4 + j] = 0;
			
		}
		m_hPos[index * 4 + 3] = 1.0f;
		m_hVel[index * 4 + 3] = 0;
		m_hForce[index * 4 + 3] = 0;
		m_numParticles++;
	}

	setArray(POSITION, m_hPos + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	//setArray(POSITION_CONTACT, m_hPos + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group);n_par_in_group));
	setArray(VELOCITY, m_hVel + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	setArray(INVERSE_MASS, m_hInvm + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	setArray(PHASE, m_hPhase + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	//setArray(SDF_arr, m_hSDF + m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	//setArray(SDF_GRADIENT, m_hSDF_gradient + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group); n_par_in_group));
	//setArray(RELATIVE_POS, m_hRelative + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group); n_par_in_group));
	setArray(FORCE, m_hForce + 4 * m_hObjectStart[m_numObjects - 1], m_numParticles - n_par_in_group, n_par_in_group);
	//setArray(ROATE, m_hRoate + 3 * 3 * (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_SIZE, m_hObjectSize, m_numObjects - 1, 1);
	setArray(OBJECT_START, m_hObjectStart + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_END, m_hObjectEnd + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_TYPE, m_hObjectType + (m_numObjects - 1), m_numObjects - 1, 1);
	setArray(OBJECT_MASS, m_hObjectMass + (m_numObjects - 1), m_numObjects - 1, 1);
}
void ParticleSystem::shaping(float deltaTime)
{
	float4 centers[MAX_OBJECT];
	float A_array[3 * 3];
	int size,start;
	float beta = m_params.beta;
	float c_creep = 0.7;
	float c_yield = 0.15;
	float c_max = 0.06;
	bool plasticity;
	for (int i = 0; i < m_numObjects; i++)
	{
		if (m_hObjectType[i] < Rigid || m_hObjectType[i] > Deformation_quad_plasticity)
			continue;
		centers[i] = Cal_Center(m_dPos_exp, m_dInvm, m_hObjectStart[i],m_hObjectSize[i],m_hObjectMass[i]);
	}
	setArray(OBJECT_CENTER, centers, 0,m_numObjects);
	for (int index_object = 0; index_object < m_numObjects; index_object++)
	{
 		if (m_hObjectType[index_object] < Rigid || m_hObjectType[index_object] > Deformation_quad_plasticity)
			continue;
		size = m_hObjectSize[index_object];
		start = m_hObjectStart[index_object];
		
		//float a = A_array[0];
		Matrix3f A;
		MatrixXf B = MatrixXf::Zero(3, 3);
		Matrix3f Q = MatrixXf::Zero(3, 3);
		Matrix3f A_a = MatrixXf::Zero(3, 3);
		Matrix3f A_qq = MatrixXf::Zero(3, 3);
		MatrixXf A_pq_quad = MatrixXf::Zero(3, 9);
		MatrixXf A_quad = MatrixXf::Zero(3, 9);
		MatrixXf A_qq_quad = MatrixXf::Zero(9, 9);
		Matrix3f Sp;
		MatrixXf Y_O = MatrixXf::Zero(3, 3);
		MatrixXf Z_O = MatrixXf::Identity(3, 3);
		MatrixXf Y = MatrixXf::Zero(3, 3);
		MatrixXf Z = MatrixXf::Identity(3, 3);
		/*Matrix3f Y_O = MatrixXf::Zero(3, 3);
		Matrix3f Z_O = MatrixXf::Identity(3, 3);
		Matrix3f Y = MatrixXf::Zero(3, 3);
		Matrix3f Z = MatrixXf::Identity(3, 3);*/
		
		
		Cal_Matrix_A(size, centers[index_object], m_dPos_exp + start * 4, m_dRelative + 4 * start, A_array);
		for (int row = 0; row < 3; row++)
		{
			for (int col = 0; col < 3; col++)
			{
				A(row, col) = A_array[row * 3 + col];
			}
		}
		
		//polar-decomposition of A
		/*Matrix3f m = A.transpose()*A;
		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		Q = svd.matrixU()*(svd.matrixV().conjugate());*/
		Y_O = A.transpose()*A;
		for (int i = 0; i < 6; i++)
		{
			Y = (Y_O + Z_O.inverse()) / 2;
			Z = (Z_O + Y_O.inverse()) / 2;
			Y_O = Y;
			Z_O = Z;

		}
		B = Y;
		Q = A * B.inverse();
		plasticity = (m_hObjectType[index_object] == Deformation_quad_plasticity || m_hObjectType[index_object] == Deformation_linear_plasticity);
		if (plasticity)
		{
			Matrix3f R;
			Matrix3f S = MatrixXf::Zero(3, 3);
			float new_A_qq[3 * 3];
			float new_A_qq_quad[9 * 9];
			MatrixXf new_A_qq_M = MatrixXf::Zero(3, 3);
			MatrixXf new_A_qq_quad_M = MatrixXf::Zero(9, 9);
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					A_qq(row, col) = m_hA_qq[index_object * 3 * 3 + row * 3 + col];
				}
			}
			A_a = A * A_qq;
			Y_O = A_a.transpose()*A_a;
			Z_O = MatrixXf::Identity(3, 3);
			for (int i = 0; i < 30; i++)
			{
				Y = (Y_O + Z_O.inverse()) / 2;
				Z = (Z_O + Y_O.inverse()) / 2;
				Y_O = Y;
				Z_O = Z;
			}
			R = A_a*Y.inverse();
			S = R.transpose() * A_a;
			MatrixXf delta = S - MatrixXf::Identity(3, 3);
			if (delta.norm() > c_yield)
			{
				for (int row = 0; row < 3; row++)
				{
					for (int col = 0; col < 3; col++)
					{
						Sp(row, col) = m_hObjectSp[index_object * 3 * 3 + row * 3 + col];
					}
				}
				
				
			
				Sp = (Matrix3f::Identity() + deltaTime * c_creep*(S - Matrix3f::Identity())) * Sp;
				Matrix3f delta_2 = Sp - Matrix3f::Identity();
				if (delta_2.norm() > c_max)
				{
					Sp = Matrix3f::Identity() + c_max * delta_2 / delta_2.norm();
					printf("!!!!!\n");
				}
					
				Sp /= pow(Sp.determinant(), 1 / 3);
				
				for (int row = 0; row < 3; row++)
				{
					for (int col = 0; col < 3; col++)
					{
						m_hObjectSp[index_object * 3 * 3 + row * 3 + col] = Sp(row, col);
					}
				}
				setArray(OBJECT_SP, m_hObjectSp + 3 * 3 * index_object, index_object, 1);
				Update_dRelative(m_dSp + index_object * 3 * 3, m_dRelative + index_object * 4, new_A_qq, new_A_qq_quad, size);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						new_A_qq_M(i, j) = new_A_qq[i * 3 + j];
					}
				}
				new_A_qq_M = new_A_qq_M.inverse();
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						m_hA_qq[index_object * 3 * 3 + i * 3 + j] = new_A_qq_M(i, j);
					}
				}
				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 9; j++)
					{
						new_A_qq_quad_M(i, j) = new_A_qq_quad[i * 9 + j];
					}
				}
				new_A_qq_quad_M = new_A_qq_quad_M.inverse();
				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 9; j++)
					{
						m_hA_qq_quad[index_object * 9 * 9 + i * 9 + j] = new_A_qq_quad_M(i, j);
					}
				}
			}
			
		}
		if (m_hObjectType[index_object] == Deformation_linear || m_hObjectType[index_object] == Deformation_linear_plasticity)
		{
			
			int index;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					A_qq(row, col) = m_hA_qq[index_object * 3 * 3 + row * 3 + col];
				}
			}
			A_a = A * A_qq;
		
			A_a /= pow(A_a.determinant(), 1 / 3);
			Q = beta*A_a + (1 - beta)*Q;
		}
		else if (m_hObjectType[index_object] == Deformation_quad || m_hObjectType[index_object] == Deformation_quad_plasticity)
		{
			float A_array[3][9];
			float A_q[9][9];
			Cal_Matrix_A_quad(size, centers[index_object], m_dPos_exp + start * 4, m_dRelative + 4 * start, &A_array[0][0]);
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					A_pq_quad(row, col) = A_array[row][col];
				}
			}
			for (int row = 0; row < 9; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					A_qq_quad(row, col) = m_hA_qq_quad[index_object * 9 * 9 + row * 9 + col];
					A_q[row][col] = A_qq_quad(row, col);
				}
			}
			A_q;
			A_quad = A_pq_quad * A_qq_quad;
			MatrixXf R_quad = MatrixXf::Zero(3, 9);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					R_quad(i, j) = Q(i, j);
				}
			}
			A_quad = beta*A_quad + (1 - beta)*R_quad;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					m_hRoate_quad[index_object * 3 * 9 + row * 9 + col] = A_quad(row, col);
				}
			}
		}
		//std::cout << Q;
		if (m_hObjectType[index_object] >= Rigid && m_hObjectType[index_object] <= Deformation_linear_plasticity)
		{
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					m_hRoate[index_object * 3 * 3 + row * 3 + col] = Q(row, col);
				}
			}
		}
	}
	setArray(ROATE, m_hRoate, 0, m_numObjects);
	setArray(ROATE_QUAD, m_hRoate_quad, 0, m_numObjects);
	Shaping_Upadate_Pos_Exp(m_dPos_exp, m_dRelative, m_dPhase, m_dCenter, m_dRoate, m_dRoate_quad, m_dObjectType,m_numParticles);
}
void ParticleSystem::shaping_BR(float deltaTime)
{
	float4 centers[MAX_OBJECT];
	float A_array[3 * 3];
	int size, start;
	float beta = m_params.beta;
	float c_creep = 0.7;
	float c_yield = 0.15;
	float c_max = 0.06;
	bool plasticity;
	/*for (int i = 0; i < m_numObjects; i++)
	{
		centers[i] = Cal_Center(m_dPos_exp, m_dInvm, m_hObjectStart[i], m_hObjectSize[i], m_hObjectMass[i]);
	}*/
	//setArray(OBJECT_CENTER, centers, 0, m_numObjects);
	Cal_Center_BR(m_dPos_exp, m_dCenter, m_dInvm, m_dObjectStart, m_dObjectSize, m_dObjectType,m_dObjectMass, m_numObjects);
	//float cc[300];
	cudaMemcpy(centers, m_dCenter, 4*MAX_OBJECT * sizeof(float), cudaMemcpyDeviceToHost);
	float A_result[3 * 3 * MAX_OBJECT];
	float A_quad_result[3 * 9 * MAX_OBJECT];
	Cal_Matrix_A_BR(m_dPos_exp, m_dCenter, m_dPhase, m_dObjectStart, m_dObjectSize, m_dRelative, m_dObjectType,A_result, m_numParticles, m_numObjects);
	Cal_Matrix_A_quad_BR(m_dPos_exp, m_dCenter, m_dPhase, m_dObjectStart, m_dObjectSize, m_dRelative, m_dObjectType, A_quad_result, m_numParticles, m_numObjects);


	/*for (int i = 0; i < 18; i++)
	{
		printf("A_result[%d] = %f\n", i, A_result[i]);
	}
	for (int i = 0; i < 54; i++)
	{
		printf("A_quad_result[%d] = %f\n", i, A_quad_result[i]);
	}*/


	for (int index_object = 0; index_object < m_numObjects; index_object++)
	{
		if (m_hObjectType[index_object] < Rigid || m_hObjectType[index_object] > Deformation_quad_plasticity)
			return;
		size = m_hObjectSize[index_object];
		start = m_hObjectStart[index_object];

		//float a = A_array[0];
		Matrix3f A;
		MatrixXf B = MatrixXf::Zero(3, 3);
		Matrix3f Q = MatrixXf::Zero(3, 3);
		Matrix3f A_a = MatrixXf::Zero(3, 3);
		Matrix3f A_qq = MatrixXf::Zero(3, 3);
		MatrixXf A_pq_quad = MatrixXf::Zero(3, 9);
		MatrixXf A_quad = MatrixXf::Zero(3, 9);
		MatrixXf A_qq_quad = MatrixXf::Zero(9, 9);
		Matrix3f Sp;
		MatrixXf Y_O = MatrixXf::Zero(3, 3);
		MatrixXf Z_O = MatrixXf::Identity(3, 3);
		MatrixXf Y = MatrixXf::Zero(3, 3);
		MatrixXf Z = MatrixXf::Identity(3, 3);
		/*Matrix3f Y_O = MatrixXf::Zero(3, 3);
		Matrix3f Z_O = MatrixXf::Identity(3, 3);
		Matrix3f Y = MatrixXf::Zero(3, 3);
		Matrix3f Z = MatrixXf::Identity(3, 3);*/
		float* A_array_BR = A_result + 3 * 3 * index_object;

		//Cal_Matrix_A(size, centers[index_object], m_dPos_exp + start * 4, m_dRelative + 4 * start, A_array);
		/*for (int i = 0; i < 9; i++)
			printf("index_obj = %d,A_array[%d] = %f\nA_array_BR[%d] =%f\n\n", index_object, i, A_array[i], i, A_array_BR[i]);*/
		for (int row = 0; row < 3; row++)
		{ 
			for (int col = 0; col < 3; col++)
			{
				A(row, col) = A_array_BR[row * 3 + col];
			}
		}

		//polar-decomposition of A
		/*Matrix3f m = A.transpose()*A;
		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		Q = svd.matrixU()*(svd.matrixV().conjugate());*/
		Y_O = A.transpose()*A;
		for (int i = 0; i < 10; i++)
		{
			Y = (Y_O + Z_O.inverse()) / 2;
			Z = (Z_O + Y_O.inverse()) / 2;
			Y_O = Y;
			Z_O = Z;

		}
		B = Y;
		Q = A * B.inverse();
		plasticity = (m_hObjectType[index_object] == Deformation_quad_plasticity || m_hObjectType[index_object] == Deformation_linear_plasticity);
		if (plasticity)
		{
			Matrix3f R;
			Matrix3f S = MatrixXf::Zero(3, 3);
			float new_A_qq[3 * 3];
			float new_A_qq_quad[9 * 9];
			MatrixXf new_A_qq_M = MatrixXf::Zero(3, 3);
			MatrixXf new_A_qq_quad_M = MatrixXf::Zero(9, 9);
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					A_qq(row, col) = m_hA_qq[index_object * 3 * 3 + row * 3 + col];
				}
			}
			A_a = A * A_qq;
			Y_O = A_a.transpose()*A_a;
			Z_O = MatrixXf::Identity(3, 3);
			for (int i = 0; i < 30; i++)
			{
				Y = (Y_O + Z_O.inverse()) / 2;
				Z = (Z_O + Y_O.inverse()) / 2;
				Y_O = Y;
				Z_O = Z;
			}
			R = A_a*Y.inverse();
			S = R.transpose() * A_a;
			MatrixXf delta = S - MatrixXf::Identity(3, 3);
			if (delta.norm() > c_yield)
			{
				for (int row = 0; row < 3; row++)
				{
					for (int col = 0; col < 3; col++)
					{
						Sp(row, col) = m_hObjectSp[index_object * 3 * 3 + row * 3 + col];
					}
				}



				Sp = (Matrix3f::Identity() + deltaTime * c_creep*(S - Matrix3f::Identity())) * Sp;
				Matrix3f delta_2 = Sp - Matrix3f::Identity();
				if (delta_2.norm() > c_max)
				{
					Sp = Matrix3f::Identity() + c_max * delta_2 / delta_2.norm();
					printf("!!!!!\n");
				}

				Sp /= pow(Sp.determinant(), 1 / 3);

				for (int row = 0; row < 3; row++)
				{
					for (int col = 0; col < 3; col++)
					{
						m_hObjectSp[index_object * 3 * 3 + row * 3 + col] = Sp(row, col);
					}
				}
				setArray(OBJECT_SP, m_hObjectSp + 3 * 3 * index_object, index_object, 1);
				Update_dRelative(m_dSp + index_object * 3 * 3, m_dRelative + index_object * 4, new_A_qq, new_A_qq_quad, size);
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						new_A_qq_M(i, j) = new_A_qq[i * 3 + j];
					}
				}
				new_A_qq_M = new_A_qq_M.inverse();
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						m_hA_qq[index_object * 3 * 3 + i * 3 + j] = new_A_qq_M(i, j);
					}
				}
				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 9; j++)
					{
						new_A_qq_quad_M(i, j) = new_A_qq_quad[i * 9 + j];
					}
				}
				new_A_qq_quad_M = new_A_qq_quad_M.inverse();
				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 9; j++)
					{
						m_hA_qq_quad[index_object * 9 * 9 + i * 9 + j] = new_A_qq_quad_M(i, j);
					}
				}
			}

		}
		if (m_hObjectType[index_object] == Deformation_linear || m_hObjectType[index_object] == Deformation_linear_plasticity)
		{

			int index;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					A_qq(row, col) = m_hA_qq[index_object * 3 * 3 + row * 3 + col];
				}
			}
			A_a = A * A_qq;

			A_a /= pow(A_a.determinant(), 1 / 3);
			Q = beta*A_a + (1 - beta)*Q;
		}
		else if (m_hObjectType[index_object] == Deformation_quad || m_hObjectType[index_object] == Deformation_quad_plasticity)
		{
			float A_array[3][9];
			float A_q[9][9];
			//Cal_Matrix_A_quad(size, centers[index_object], m_dPos_exp + start * 4, m_dRelative + 4 * start, &A_array[0][0]);
			float* A_quad_array_BR = A_quad_result + 3 * 9 * index_object;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					A_pq_quad(row, col) = A_quad_array_BR[9 * row + col];
				}
			}
			for (int row = 0; row < 9; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					A_qq_quad(row, col) = m_hA_qq_quad[index_object * 9 * 9 + row * 9 + col];
					A_q[row][col] = A_qq_quad(row, col);
				}
			}
			A_q;
			A_quad = A_pq_quad * A_qq_quad;
			MatrixXf R_quad = MatrixXf::Zero(3, 9);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					R_quad(i, j) = Q(i, j);
				}
			}
			A_quad = beta*A_quad + (1 - beta)*R_quad;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 9; col++)
				{
					m_hRoate_quad[index_object * 3 * 9 + row * 9 + col] = A_quad(row, col);
				}
			}
		}
		//std::cout << Q;
 		if (m_hObjectType[index_object] >= Rigid && m_hObjectType[index_object] <= Deformation_linear_plasticity)
		{
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					m_hRoate[index_object * 3 * 3 + row * 3 + col] = Q(row, col);
				}
			}
		}
	}
	setArray(ROATE, m_hRoate, 0, m_numObjects);
 	setArray(ROATE_QUAD, m_hRoate_quad, 0, m_numObjects);
	Shaping_Upadate_Pos_Exp(m_dPos_exp, m_dRelative, m_dPhase, m_dCenter, m_dRoate, m_dRoate_quad, m_dObjectType, m_numParticles);
}
void ParticleSystem::renderMesh()
{
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertexVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertexcolorVBO);
	glColorPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_NorVBO);
	glNormalPointer(GL_FLOAT, 0, 0);
	glEnableClientState(GL_NORMAL_ARRAY);


	

	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, m_indexVBO);
	/*if (m_colorVBO)
	{
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
	}*/
	


	/*  draw sphere in first row, first column
	*  diffuse reflection only; no ambient or specular
	*/
	//glDisableClientState(GL_COLOR_ARRAY);

	//const void * device = mapGLBufferObject(&m_cuda_vertexvbo_resource);
	////copyArrayFromDevice(mp_vertexXYZ, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	//cudaMemcpy(m_ver, device, m_totalConnectedPoints * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	//unmapGLBufferObject(m_cuda_vertexvbo_resource);


	float3 v1, v2, v3;
	uint index_1, index_2, index_3;
	
	glDrawElements(GL_TRIANGLES, m_numFaces * 3, GL_UNSIGNED_INT, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisable(GL_COLOR_MATERIAL);

}