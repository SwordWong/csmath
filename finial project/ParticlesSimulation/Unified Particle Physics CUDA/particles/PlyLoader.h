#ifndef PLYREADER_H_
#define PLYREADER_H_

//#include <GL/glut.h>
//#include <GL/glu.h>
//#include <GL/gl.h>
#include <vector>
//#include <helper_math.h>
#include <iostream>
//#include "vector_types.h"
#include <nvVector.h>
using namespace std;
using namespace::nv;
struct SModelData
{
	vector <float> vecFaceTriangles; // = face * 9
	vector <float> vecFaceTriangleColors; // = face * 9
	vector <float> vecNormals; // = face * 9
	int iTotalConnectedTriangles;
};

class CPLYLoader
{
public:
	CPLYLoader();
	int LoadModel(char *filename);
	void Draw();
	float* mp_vertexXYZ;
	float* mp_vertexNorm;
	float* mp_vertexRGB;

	unsigned int* mp_face;
	vec3<float>* m_renderPos;
	//float* mp_vertexNorm;
	//float* mp_vertexRGB;
	int m_totalConnectedQuads;
	int m_totalConnectedPoints;
	int m_totalFaces;
	SModelData m_ModelData;
private:
	
};

#endif