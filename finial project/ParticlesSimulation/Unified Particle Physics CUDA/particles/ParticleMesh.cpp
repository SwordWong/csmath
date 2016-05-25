#include <ParticleMesh.h>
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
#include "particleSystem.cuh"
uint
ParticleMesh::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}
ParticleMesh::~ParticleMesh()
{
	/*delete[] par_list;
	delete[] sdf_list;
	delete[] mp_vertexXYZ;
	delete[] Relative_Vertex;
	delete[] FaceVerInd;*/
	release();
}


float ParticleMesh::CalScale(nv::vec3<double> MaxB, nv::vec3<double> MinB, float r, int size)
{
	vec3<double> bb = MaxB - MinB;
	float Length = MAX(MAX(bb[0], bb[1]), bb[2]);
	float step = Length / size;
	return 2 * r / step;
}


void ParticleMesh::calculate(LevelsetCollider &LC, int size, float r, float mass)
 {
	
	float scale = 2 * r / (LC.gridStep*(LC.gridSize.x - 4) / size);
	vec3<double> center = vec3<double>(0, 0, 0);
	vec3<double> MaxB = LC.MaxBoundary;
	vec3<double> MinB = LC.MinBoundary;
	scale = CalScale(MaxB, MinB, r, size);
	int index_par = 0;
	par_list = new triple[20000];
	sdf_list = new SDF[20000];
	float step = r * 2 / scale;
	

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k <size; k++)
			{
				vec3<int> gridPos = vec3<int>(i, j, k);
				vec3<double> pos = vec3<double>(i + 0.5, j + 0.5, k + 0.5) * (2 * r / scale) + MinB;
				int gridInd = dot(gridPos, LC.gridSizeOffset);
				double sdf = 0;
				LC.getGridDist(pos, sdf);
				if (sdf < 0 && sdf > -r * 2 / scale * 2)
				{
					vec3<double> pos_par = pos - MinB;// -vec3<double>(0.5, 0.5, 0.5) * (2 * r / scale);

					center += mass*pos_par * scale;
					par_list[index_par].x[0] = pos_par.x * scale;
					par_list[index_par].x[1] = pos_par.y * scale;
					par_list[index_par].x[2] = pos_par.z * scale;

					sdf_list[index_par].sdf = sdf * scale;
					vec3<double> gradient;
					LC.checkCollision(pos, sdf, gradient, -1);
					sdf_list[index_par].gradient[0] = gradient.x;
					sdf_list[index_par].gradient[1] = gradient.y;
					sdf_list[index_par].gradient[2] = gradient.z;
					index_par++;
				}
			}
		}
	}


	center /= index_par*mass;
	float3 center_f3 = make_float3(center[0], center[1], center[2]);
	m_totalParticles = index_par;
	m_totalConnectedPoints = LC.meshModel->m_totalConnectedPoints;
	m_totalFaces = LC.meshModel->m_totalFaces;
	float3 vertex,relative;
	mp_vertexXYZ = new float[m_totalConnectedPoints * 3];
	Relative_Vertex = new float3[m_totalConnectedPoints];
	mp_vertexNorm = new float[m_totalConnectedPoints * 3];
	mp_vertexRGB = new float[m_totalConnectedPoints * 3];
	for (int i = 0; i < m_totalConnectedPoints; i++)
	{
		vertex.x = LC.meshModel->mp_vertexXYZ[i * 3];
		vertex.y = LC.meshModel->mp_vertexXYZ[i * 3 + 1];
		vertex.z = LC.meshModel->mp_vertexXYZ[i * 3 + 2];
		//vertex *= scale;
		vertex -= make_float3(MinB[0], MinB[1], MinB[2]);// +make_float3(0.5, 0.5, 0.5) * (2 * r / scale);
		vertex *= scale;
		relative = vertex - center_f3;
		*(float3*)(mp_vertexXYZ + i * 3) = vertex;


		mp_vertexRGB[i * 3] = LC.meshModel->mp_vertexRGB[i * 3];
		mp_vertexRGB[i * 3 + 1] = LC.meshModel->mp_vertexRGB[i * 3 + 1];
		mp_vertexRGB[i * 3 + 2] = LC.meshModel->mp_vertexRGB[i * 3 + 2];

		mp_vertexNorm[i * 3] = 0; //LC.meshModel->mp_vertexNorm[i * 3];
		mp_vertexNorm[i * 3 + 1] = 0; //LC.meshModel->mp_vertexNorm[i * 3 + 1];
		mp_vertexNorm[i * 3 + 2] = 0;// LC.meshModel->mp_vertexNorm[i * 3 + 2];


		Relative_Vertex[i] = relative;
	}

	FaceVerInd = new uint[3 * m_totalFaces];
	//memcpy(FaceVerInd, LC.meshModel->mp_face, 9 * m_totalFaces * sizeof(uint));
	float3  n;
	for (int i = 0; i < m_totalFaces; i++)
	{
		uint i_1 = FaceVerInd[3 * i] = LC.meshModel->mp_face[3 * i];
		uint i_2 = FaceVerInd[3 * i + 1] = LC.meshModel->mp_face[3 * i + 1];
		uint i_3 = FaceVerInd[3 * i + 2] = LC.meshModel->mp_face[3 * i + 2];

		float3 v1 = make_float3(mp_vertexXYZ[i_1 * 3], mp_vertexXYZ[i_1 * 3 + 1], mp_vertexXYZ[i_1 * 3 + 2]);
		float3 v2 = make_float3(mp_vertexXYZ[i_2 * 3], mp_vertexXYZ[i_2 * 3 + 1], mp_vertexXYZ[i_2 * 3 + 2]);
		float3 v3 = make_float3(mp_vertexXYZ[i_3 * 3], mp_vertexXYZ[i_3 * 3 + 1], mp_vertexXYZ[i_3 * 3 + 2]);

		float3 e12 = v2 - v1;
		float3 e13 = v3 - v1;
		n = cross(e12, e13);
		n /= length(n);
		mp_vertexNorm[i_1 * 3] += n.x;
		mp_vertexNorm[i_1 * 3 + 1] += n.y;
		mp_vertexNorm[i_1 * 3 + 2] += n.z;

		//atomicAdd(&num_tri_per_point[i_1], 1);

		float3 e21 = -e12;
		float3 e23 = v3 - v2;
		n = cross(e23, e21);
		n /= length(n);
		mp_vertexNorm[i_2 * 3] += n.x;
		mp_vertexNorm[i_2 * 3 + 1] += n.y;
		mp_vertexNorm[i_2 * 3 + 2] += n.z;

		//atomicAdd(&num_tri_per_point[i_2], 1);

		float3 e31 = -e13;
		float3 e32 = -e23;
		n = cross(e31, e32);
		n /= length(n);
		mp_vertexNorm[i_3 * 3] += n.x;
		mp_vertexNorm[i_3 * 3 + 1] += n.y;
		mp_vertexNorm[i_3 * 3 + 2] += n.z;
	}
	for (int i = 0; i < m_totalConnectedPoints; i++)
	{
		

		float3 n = make_float3(mp_vertexNorm[i * 3], mp_vertexNorm[i * 3 + 1], mp_vertexNorm[i * 3 + 2]);

		mp_vertexNorm[i * 3] /= length(n); 
		mp_vertexNorm[i * 3 + 1] /= length(n);
		mp_vertexNorm[i * 3 + 2] /= length(n);
	}
	m_vertexVbo = createVBO(m_totalConnectedPoints * 3 * sizeof(float));
	registerGLBufferObject(m_vertexVbo, &m_cuda_vertexvbo_resource);

	m_indexVBO = createVBO(3 * m_totalFaces * sizeof(uint));
	registerGLBufferObject(m_indexVBO, &m_cuda_indexvbo_resource);

	m_colorVBO = createVBO(m_totalConnectedPoints * 3 * sizeof(float));
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

	m_NorVBO = createVBO(m_totalConnectedPoints * 3 * sizeof(float));
	registerGLBufferObject(m_NorVBO, &m_cuda_normalvbo_resource);

	unregisterGLBufferObject(m_cuda_vertexvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexVbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_totalConnectedPoints * 3 * sizeof(float), mp_vertexXYZ);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_vertexVbo, &m_cuda_vertexvbo_resource);

	unregisterGLBufferObject(m_cuda_indexvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_indexVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, 3 * m_totalFaces * sizeof(uint), FaceVerInd);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_indexVBO, &m_cuda_indexvbo_resource);

	unregisterGLBufferObject(m_cuda_colorvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_totalConnectedPoints * 3 * sizeof(float), mp_vertexRGB);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);



	unregisterGLBufferObject(m_cuda_normalvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_NorVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_totalConnectedPoints * 3 * sizeof(float), mp_vertexNorm);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_NorVBO, &m_cuda_normalvbo_resource);

	allocateArray((void**)&dRelative_Vertex, m_totalConnectedPoints * 3 * sizeof(float));
	copyArrayToDevice(dRelative_Vertex, Relative_Vertex, 0, m_totalConnectedPoints * 3 * sizeof(float));
	hasMesh = true;
}
void ParticleMesh::renderer()
{

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertexVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	/*glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
	glColorPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);*/

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
	


	

	const void * device = mapGLBufferObject(&m_cuda_vertexvbo_resource);
	//copyArrayFromDevice(mp_vertexXYZ, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
	cudaMemcpy(mp_vertexXYZ, device, m_totalConnectedPoints * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	unmapGLBufferObject(m_cuda_vertexvbo_resource);


	

	
	
	glDrawElements(GL_TRIANGLES, m_totalFaces * 3, GL_UNSIGNED_INT, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisable(GL_COLOR_MATERIAL);
}
void ParticleMesh::release()
{
	if (!par_list)
		delete[] par_list;
	if (!sdf_list)
		delete[] sdf_list;
	if (!mp_vertexXYZ)
		delete[] mp_vertexXYZ;
	if (!Relative_Vertex)
		delete[] Relative_Vertex;
	if (!FaceVerInd)
		delete[] FaceVerInd;
	if (!mp_vertexRGB)
		delete[] mp_vertexRGB;
	if (!mp_vertexNorm)
		delete[] mp_vertexNorm;
	if (!dRelative_Vertex)
		freeArray(dRelative_Vertex);
	unregisterGLBufferObject(m_cuda_indexvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_indexVBO);

	unregisterGLBufferObject(m_cuda_normalvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_NorVBO);

	unregisterGLBufferObject(m_cuda_colorvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_colorVBO);

	unregisterGLBufferObject(m_cuda_vertexvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_vertexVbo);
}