#ifndef PARTICLEMESH	
#define PARTICLEMESH
#include <vector>
#include <iostream>
#include <Object.h>
//#include <LevelsetCollider.h>

//#include "vector_types.h"
//#include <nvVector.h>
#include <helper_math.h>
#include<tmp.h>
using namespace std;
//using namespace::nv;
class ParticleMesh
{
public:
	bool hasMesh;
	float* mp_vertexXYZ;
	float* mp_vertexNorm;
	float* mp_vertexRGB;
	int m_totalConnectedQuads;
	int m_totalConnectedPoints;
	int m_totalFaces;
	int m_totalParticles;
	float3 *FaceTriangles; // = face * 9
	uint *FaceVerInd;
	float3 *Relative_Vertex;
	triple *par_list;
	SDF *sdf_list;
	uint   m_vertexVbo;            // vertex buffer object for particle positions
	uint   m_indexVBO;          
	uint   m_colorVBO;
	uint   m_NorVBO;
	uint   m_NorOriginalVBO;
	float *m_cudaVerVBO;        // these are the CUDA deviceMem Pos
	float *m_cudaIndexVBO;      // these are the CUDA deviceMem Color

	float* dRelative_Vertex;
	struct cudaGraphicsResource *m_cuda_vertexvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_indexvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_normalvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_normal_original_vbo_resource;

	ParticleMesh()
	{
		m_totalConnectedPoints = 0;
		m_totalConnectedQuads = 0;
		m_totalFaces = 0;
		m_totalParticles = 0;
		mp_vertexXYZ = NULL;
		hasMesh = false;
		FaceTriangles = NULL;
	}
	~ParticleMesh();
	uint createVBO(uint size);
	void calculate(LevelsetCollider &LC, int size, float r, float mass);
	void renderer();
	void release();
	float CalScale(nv::vec3<double> MaxB, nv::vec3<double> MinB, float r, int size);
};
#endif