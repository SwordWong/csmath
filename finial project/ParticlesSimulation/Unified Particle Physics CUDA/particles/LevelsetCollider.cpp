#include "LevelsetCollider.h"

#include <float.h>

#include <omp.h>

//#include <QFile>
//#include <QTextStream>
#include<helper_image.h>
#include <vector>

using namespace nv;

vec3<double> pointSegmentDist(const vec3<double> &x0, const vec3<double> &x1, const vec3<double> &x2)
{
	const vec3<double> dx = x2 - x1;
	const double coord = MAX(MIN(dot((x2 - x0), dx) / dot(dx,dx), 1.), 0.);
	return (x1 * coord + x2 * (1. - coord)) - x0;
}

vec3<double> pointTriangleDist(const vec3<double> &x0, const vec3<double> &x1, const vec3<double> &x2, const vec3<double> &x3)
{
	const vec3<double> x13 = x1 - x3;
	const vec3<double> x23 = x2 - x3;
	const vec3<double> x03 = x0 - x3;
	const double m13 = dot(x13,x13);
	const double m23 = dot(x23,x23);
	const double d = dot(x13,x23);
	const double invdet = 1. / MAX(m13 * m23 - d * d, 1e-30);
	const double a = dot(x13,x03);
	const double b = dot(x23,x03);
	double w23 = invdet * (m23 * a - d * b);
	double w31 = invdet * (m13 * b - d * a);
	double w12 = 1. - w23 - w31;
	if (w23 >= 0. && w31 >= 0. && w12 >= 0.)
		return (x1 * w23 + x2 * w31 + x3 * w12) - x0;
	else if (w23 > 0.){
		const vec3<double> distVec1 = pointSegmentDist(x0, x1, x2);
		const vec3<double> distVec2 = pointSegmentDist(x0, x1, x3);
		return (dot(distVec1, distVec1) < dot(distVec2, distVec2)) ? distVec1 : distVec2;
	}
	else if (w31 > 0.){
		const vec3<double> distVec1 = pointSegmentDist(x0, x1, x2);
		const vec3<double> distVec2 = pointSegmentDist(x0, x2, x3);
		return (dot(distVec1, distVec1) < dot(distVec2, distVec2)) ? distVec1 : distVec2;
	}
	else{
		const vec3<double> distVec1 = pointSegmentDist(x0, x1, x3);
		const vec3<double> distVec2 = pointSegmentDist(x0, x2, x3);
		return (dot(distVec1, distVec1) < dot(distVec2, distVec2)) ? distVec1 : distVec2;
	}
}

char triangleOrient(const vec2<double>& p1, const vec2<double>& p2, double &area)
{
	area = p1[1] * p2[0] - p1[0] * p2[1];
	return (area > 0.) ? 1 : -1;
}

bool pointTriangle2D(const vec2<double>& p0, const vec2<double>& p1, const vec2<double>& p2, const vec2<double>& p3, vec3<double>& coord)
{
	const vec2<double> p01 = p1 - p0;
	const vec2<double> p02 = p2 - p0;
	const vec2<double> p03 = p3 - p0;
	const char signa = triangleOrient(p02, p03, coord[0]);
	if (triangleOrient(p03, p01, coord[1]) != signa || triangleOrient(p01, p02, coord[2]) != signa)
		return false;
	coord /= coord[0] + coord[1] + coord[2];
	return true;
}

void solvePhi(double p, double q, double r, double& s)
{
	const double minV = MIN(MIN(p, q), r);
	const double maxV = MAX(MAX(p, q), r);
	const double midV = p + q + r - minV - maxV;
	double d = minV + 1.;
	if (d > maxV)
		d = (minV + midV + maxV + sqrt(3. - 2. * (minV * minV + midV * midV + maxV * maxV - minV * midV - minV * maxV - midV * maxV))) / 3.;
	else if (d > midV)
		d = (minV + midV + sqrt(2. - (minV - midV) * (minV - midV))) / 2.;
	if (d < s)
		s = d;
}

LevelsetCollider::LevelsetCollider()
{
	gridOrigin = vec3<double>(0., 0., 0.);
	gridSize = vec3<int>(0, 0, 0);
	gridSizeOffset = vec3<int>(0, 0, 0);
	gridStep = 0.;
	gridStepInv = 0.;

	testThreshold = 0.;

	gridOffset = 0.;
	gridMargin = 0.;

	distGrid = NULL;
}

LevelsetCollider::~LevelsetCollider()
{
	delete[] distGrid;
}

//bool LevelsetCollider::load(QString fileName, double offset)
//{
//	QFile levelsetFile(fileName);
//	if (!levelsetFile.open(QIODevice::ReadOnly))
//		return false;
//	levelsetFile.read((char*)&gridOrigin, sizeof(vec3<double>));
//	levelsetFile.read((char*)&gridSize, sizeof(vec3<int>));
//	levelsetFile.read((char*)&gridStep, sizeof(double));
//	gridStepInv = 1. / gridStep;
//	const int cellNum = gridSize[0] * gridSize[1] * gridSize[2];
//	delete[] distGrid;
//	distGrid = new double[cellNum];
//	levelsetFile.read((char*)distGrid, sizeof(double) * cellNum);
//	levelsetFile.close();
//	gridSizeOffset = vec3<int>(1, gridSize[0], gridSize[0] * gridSize[1]);
//	testThreshold = sqrt(3.) * gridStep;
//	gridOffset = offset;
//
//	bbMin = gridOrigin - vec3<double>(offset, offset, offset);
//	bbMax = gridOrigin + vec3<double>(gridSize) * gridStep + vec3<double>(offset, offset, offset);
//	bbMin = bbMin * adjustScale + adjustTranslate;
//	bbMax = bbMax * adjustScale + adjustTranslate;
//
//	return true;
//}
//
//bool LevelsetCollider::save(QString fileName)
//{
//	QFile levelsetFile(fileName);
//	if (!levelsetFile.open(QIODevice::WriteOnly))
//		return false;
//	levelsetFile.write((char*)&gridOrigin, sizeof(vec3<double>));
//	levelsetFile.write((char*)&gridSize, sizeof(vec3<int>));
//	levelsetFile.write((char*)&gridStep, sizeof(double));
//	const int cellNum = gridSize[0] * gridSize[1] * gridSize[2];
//	levelsetFile.write((char*)distGrid, sizeof(double) * cellNum);
//	levelsetFile.close();
//	return true;
//}

void LevelsetCollider::setMeshModel(CPLYLoader& model)
{
	meshModel = &model;
}

void LevelsetCollider::calculate(int size, double offset, double margin)
{
#define EXACTBAND 2

	// Create grid structures

	gridOffset = offset;
	gridMargin = margin;
	vec3<double> gridMin = vec3<double>(DBL_MAX, DBL_MAX, DBL_MAX);
	vec3<double> gridMax = vec3<double>(-DBL_MAX, -DBL_MAX, -DBL_MAX);
	const int faceNum = meshModel->m_totalFaces;
	const vec3<float>* const faces = meshModel->m_renderPos;
	for (int faceI = 0; faceI < faceNum; ++faceI){
		for (int vertexI = 0; vertexI < 3; ++vertexI){
			const vec3<double>& vertex = vec3<double>(faces[faceI * 3 + vertexI][0], faces[faceI * 3 + vertexI][1], faces[faceI * 3 + vertexI][2]);
			for (int dimI = 0; dimI < 3; ++dimI){
				gridMin[dimI] = MIN(gridMin[dimI], vertex[dimI]);
				gridMax[dimI] = MAX(gridMax[dimI], vertex[dimI]);
			}
		}
	}
	MaxBoundary = gridMax;
	MinBoundary = gridMin;
	gridMin -= vec3<double>(offset + margin, offset + margin, offset + margin);
	gridMax += vec3<double>(offset + margin, offset + margin, offset + margin);
	const vec3<double> bb = gridMax - gridMin;
	gridStep = pow((bb[0] * bb[1] * bb[2]) / (size * size * size), 1. / 3.);
	gridStepInv = 1. / gridStep;
	testThreshold = sqrt(3.) * gridStep;
	for (int dimI = 0; dimI < 3; ++dimI){
		gridSize[dimI] = int(ceil(bb[dimI] * gridStepInv)) + 4;
		gridOrigin[dimI] = gridMin[dimI] - (gridSize[dimI] * gridStep - bb[dimI]) * 0.5 - 1e-6 * gridStep;
	}
	const int cellNum = gridSize[0] * gridSize[1] * gridSize[2];
	gridSizeOffset = vec3<int>(1, gridSize[0], gridSize[0] * gridSize[1]);
	delete[] distGrid;
	distGrid = new double[cellNum];
	int* const intersectGrid = new int[cellNum];
	char* const boundaryGrid = new char[cellNum];
	for (int cellI = 0; cellI < cellNum; ++cellI)
		distGrid[cellI] = DBL_MAX;
	memset(intersectGrid, 0, sizeof(int) * cellNum);
	memset(boundaryGrid, 0, sizeof(char) * cellNum);

	// Calculate exact boundary distances and signs

	for (int faceI = 0; faceI < faceNum; ++faceI){
		//system("cls");
		printf("Face %d of %d\n", faceI, faceNum);
		vec3<double> triCoord[3];
		vec3<int> triBB[2];
		for (int vertexI = 0; vertexI < 3; ++vertexI)
			triCoord[vertexI] = (vec3<double>(faces[faceI * 3 + vertexI].x, faces[faceI * 3 + vertexI].y, faces[faceI * 3 + vertexI].z) - gridOrigin) * gridStepInv;
		for (int dimI = 0; dimI < 3; ++dimI){
			triBB[0][dimI] = int(floor(MIN(MIN(triCoord[0][dimI], triCoord[1][dimI]), triCoord[2][dimI]))) - EXACTBAND;
			triBB[1][dimI] = int(ceil(MAX(MAX(triCoord[0][dimI], triCoord[1][dimI]), triCoord[2][dimI]))) + EXACTBAND + 1;
			triBB[0][dimI] = MAX(MIN(triBB[0][dimI], gridSize[dimI] - 1), 0);
			triBB[1][dimI] = MAX(MIN(triBB[1][dimI], gridSize[dimI] - 1), 0);
		}
		for (int kI = triBB[0][2]; kI <= triBB[1][2]; ++kI){
			for (int jI = triBB[0][1]; jI <= triBB[1][1]; ++jI){
				for (int iI = triBB[0][0]; iI <= triBB[1][0]; ++iI){
					const int gridInd = dot(vec3<int>(iI, jI, kI), gridSizeOffset);
					const vec3<double> cellPos = vec3<double>(iI, jI, kI) * gridStep + gridOrigin;
					vec3<double> distVec = pointTriangleDist(cellPos,
						vec3<double>(faces[faceI * 3].x, faces[faceI * 3].y, faces[faceI * 3].z),
						vec3<double>(faces[faceI * 3 + 1].x, faces[faceI * 3 + 1].y, faces[faceI * 3 + 1].z),
						vec3<double>(faces[faceI * 3 + 2].x, faces[faceI * 3 + 2].y, faces[faceI * 3 + 2].z));
					const double dist2 = dot(distVec, distVec);
					if (dist2 < distGrid[gridInd]){
						distGrid[gridInd] = dist2;
						boundaryGrid[gridInd] = 1;
					}
				}
			}
		}
		vec3<int> triCountBBMin;
		vec3<int> triCountBBMax;
		for (int dimI = 0; dimI < 3; ++dimI){
			triCountBBMin[dimI] = int(ceil(MIN(MIN(triCoord[0][dimI], triCoord[1][dimI]), triCoord[2][dimI])));
			triCountBBMax[dimI] = int(floor(MAX(MAX(triCoord[0][dimI], triCoord[1][dimI]), triCoord[2][dimI])));
			triCountBBMin[dimI] = MAX(MIN(triCountBBMin[dimI], gridSize[dimI] - 1), 0);
			triCountBBMax[dimI] = MAX(MIN(triCountBBMax[dimI], gridSize[dimI] - 1), 0);
		}
		for (int kI = triCountBBMin[2]; kI <= triCountBBMax[2]; ++kI){
			for (int jI = triCountBBMin[1]; jI <= triCountBBMax[1]; ++jI){
				vec3<double> vertexInd;
				if (pointTriangle2D(vec2<double>(jI, kI),
					vec2<double>(triCoord[0][1], triCoord[0][2]),
					vec2<double>(triCoord[1][1], triCoord[1][2]),
					vec2<double>(triCoord[2][1], triCoord[2][2]),
					vertexInd)){
					const double fi = dot(vertexInd,vec3<double>(triCoord[0][0], triCoord[1][0], triCoord[2][0]));
					const int iInterval = int(ceil(fi));
					if (iInterval < 0)
						++intersectGrid[dot(vec3<int>(0, jI, kI),gridSizeOffset)];
					else if (iInterval < gridSize[0])
						++intersectGrid[dot(vec3<int>(iInterval, jI, kI),gridSizeOffset)];
				}
			}
		}
	}
	printf("\n");
	for (int kI = 0; kI < gridSize[2]; ++kI){
		for (int jI = 0; jI < gridSize[1]; ++jI){
			for (int iI = 0; iI < gridSize[0]; ++iI){
				const int gridInd = dot(vec3<int>(iI, jI, kI),gridSizeOffset);
				if (distGrid[gridInd] > 0.)
					distGrid[gridInd] = sqrt(distGrid[gridInd]);
			}
		}
	}

	// Sweep to all empty cells

	for (int iterI = 0; iterI < 2; iterI++){
		for (int order = 0; order < 8; ++order){
			const int di = (order / 4 % 2) ? -1 : 1;
			const int dj = (order / 2 % 2) ? -1 : 1;
			const int dk = (order / 1 % 2) ? -1 : 1;
			const int iBegin = (di == 1) ? 1 : gridSize[0] - 2;
			const int jBegin = (dj == 1) ? 1 : gridSize[1] - 2;
			const int kBegin = (dk == 1) ? 1 : gridSize[2] - 2;
			const int iEnd = (di == 1) ? gridSize[0] - 1 : 0;
			const int jEnd = (dj == 1) ? gridSize[1] - 1 : 0;
			const int kEnd = (dk == 1) ? gridSize[2] - 1 : 0;
			for (int k = kBegin; k != kEnd; k += dk){
				for (int j = jBegin; j != jEnd; j += dj){
					for (int i = iBegin; i != iEnd; i += di){
						const int gridInd = dot(vec3<int>(i, j, k),gridSizeOffset);
						if (boundaryGrid[gridInd] == 0)
							solvePhi(
								distGrid[gridInd - di],
								distGrid[gridInd - dj * gridSizeOffset[1]],
								distGrid[gridInd - dk * gridSizeOffset[2]],
								distGrid[gridInd]);
					}
				}
			}
		}
	}

	// Apply signs

#pragma omp parallel for
	for (int kI = 1; kI < gridSize[2] - 1; ++kI){
		for (int jI = 1; jI < gridSize[1] - 1; ++jI){
			int count = 0;
			for (int iI = 1; iI < gridSize[0] - 1; ++iI){
				const int gridInd = dot(vec3<int>(iI, jI, kI),gridSizeOffset);
				count += intersectGrid[gridInd];
				if (count & 1)
					distGrid[gridInd] = -distGrid[gridInd];
				distGrid[gridInd] -= offset;
			}
		}
	}

	delete[] intersectGrid;
	delete[] boundaryGrid;

	//bbMin = gridOrigin - vec3<double>(offset, offset, offset);
	//bbMax = gridOrigin + vec3<double>(gridSize) * gridStep + vec3<double>(offset, offset, offset);
	//bbMin = bbMin * adjustScale + adjustTranslate;
	//bbMax = bbMax * adjustScale + adjustTranslate;
}

bool LevelsetCollider::testCollision(const vec3<double>& pos, int index)
{
	/*if (!checkBB(pos))
		return false;*/

	const vec3<double> testPos = pos;

	const vec3<double> gridPos = (testPos - gridOrigin) * gridStepInv;
	if (gridPos[0] < 1. || gridPos[0] >= double(gridSize[0] - 2) ||
		gridPos[1] < 1. || gridPos[1] >= double(gridSize[1] - 2) ||
		gridPos[2] < 1. || gridPos[2] >= double(gridSize[2] - 2))
		return false;

	const vec3<int> gridPosI = vec3<int>(int(gridPos[0]), int(gridPos[1]), int(gridPos[2]));
	const int gridInd = dot(gridPosI,gridSizeOffset);
	if (distGrid[gridInd] > testThreshold)
		return false;

	const int gridInd1 = gridInd + 1;
	const vec3<double> gridPosF = vec3<double>(gridPos[0] - gridPosI[0], gridPos[1] - gridPosI[1], gridPos[2] - gridPosI[2]);
	const vec3<double> gridPosFInv = vec3<double>(1., 1., 1.) - gridPosF;
	const double zTerm0 =
		distGrid[gridInd] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm1 =
		distGrid[gridInd + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm2 =
		distGrid[gridInd1] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm3 =
		distGrid[gridInd1 + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double yTerm0 =
		zTerm0 * gridPosFInv[1] +
		zTerm1 * gridPosF[1];
	const double yTerm1 =
		zTerm2 * gridPosFInv[1] +
		zTerm3 * gridPosF[1];
	double dist =
		yTerm0 * gridPosFInv[0] +
		yTerm1 * gridPosF[0];
	return dist < 0.;
}

bool LevelsetCollider::testCollision(const vec3<double>& pos1, const vec3<double>& pos2, int index)
{
	vec3<double> posMin, posMax;
	posMin[0] = MIN(pos1[0], pos2[0]);
	posMin[1] = MIN(pos1[1], pos2[1]);
	posMin[2] = MIN(pos1[2], pos2[2]);
	posMax[0] = MAX(pos1[0], pos2[0]);
	posMax[1] = MAX(pos1[1], pos2[1]);
	posMax[2] = MAX(pos1[2], pos2[2]);
	///*if (!checkBB(posMin, posMax))
	//	return false;*/

	//else
	//	return true;

	vec3<double> dispVec = pos2 - pos1;
	double disp = dot(dispVec, dispVec);
	if (disp < 1e-10)
		return false;

	dispVec /= disp;
	vec3<double> curPos = pos1;
	double accLen = 0.;

	double dist1, dist2;
	getGridDist(pos1, dist1);
	getGridDist(pos2, dist2);
	double baseDist = MIN(MIN(dist1, dist2), 0.);

	int count = 0;
	while (true){
		double dist;
		getGridDist(curPos, dist);
		if (dist < baseDist){
			//printf("get\n");
			return true;
		}
		dist = disp * 0.01;// MAX(dist, disp * 0.02);
		curPos += dispVec * dist;
		accLen += dist;
		if (accLen > disp)
			return false;
	}
}

bool LevelsetCollider::checkCollision(const vec3<double>& pos, double& dist, vec3<double>& normal, int index)
{
	dist = DBL_MAX;
	/*if (!checkBB(pos))
		return false;*/

	const vec3<double> testPos = pos;

	const vec3<double> gridPos = (testPos - gridOrigin) * gridStepInv;
	if (gridPos[0] < 1. || gridPos[0] >= double(gridSize[0] - 2) ||
		gridPos[1] < 1. || gridPos[1] >= double(gridSize[1] - 2) ||
		gridPos[2] < 1. || gridPos[2] >= double(gridSize[2] - 2))
		return false;

	const vec3<int> gridPosI = vec3<int>(int(gridPos[0]), int(gridPos[1]), int(gridPos[2]));
	const int gridInd = dot(gridPosI,gridSizeOffset);
	if (distGrid[gridInd] > testThreshold)
		return false;

	const int gridInd1 = gridInd + 1;
	const vec3<double> gridPosF = vec3<double>(gridPos[0] - gridPosI[0], gridPos[1] - gridPosI[1], gridPos[2] - gridPosI[2]);
	const vec3<double> gridPosFInv = vec3<double>(1., 1., 1.) - gridPosF;
	const double zTerm0 =
		distGrid[gridInd] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm1 =
		distGrid[gridInd + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm2 =
		distGrid[gridInd1] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm3 =
		distGrid[gridInd1 + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double yTerm0 =
		zTerm0 * gridPosFInv[1] +
		zTerm1 * gridPosF[1];
	const double yTerm1 =
		zTerm2 * gridPosFInv[1] +
		zTerm3 * gridPosF[1];
	dist = 
		yTerm0 * gridPosFInv[0] +
		yTerm1 * gridPosF[0];
	if (dist >= 0.)
		return false;
	dist = -dist;

	const double distLeft = yTerm0;
	const double distRight = yTerm1;
	const double distBottom =
		zTerm0 * gridPosFInv[0] +
		zTerm2 * gridPosF[0];
	const double distUp =
		zTerm1 * gridPosFInv[0] +
		zTerm3 * gridPosF[0];
	const double distNear = (
		distGrid[gridInd] * gridPosFInv[1] +
		distGrid[gridInd + gridSizeOffset[1]] * gridPosF[1]) * gridPosFInv[0] + (
		distGrid[gridInd1] * gridPosFInv[1] +
		distGrid[gridInd1 + gridSizeOffset[1]] * gridPosF[1]) * gridPosF[0];
	const double distFar = (
		distGrid[gridInd + gridSizeOffset[2]] * gridPosFInv[1] +
		distGrid[gridInd + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[1]) * gridPosFInv[0] + (
		distGrid[gridInd1 + gridSizeOffset[2]] * gridPosFInv[1] +
		distGrid[gridInd1 + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[1]) * gridPosF[0];
	normal = vec3<double>(distRight - distLeft, distUp - distBottom, distFar - distNear);

	if (normal[0] == 0. && normal[1] == 0. && normal[2] == 0.)
		return false;

	normal = normalize(normal);
	return true;
}

bool LevelsetCollider::checkCollision(const vec3<double>& pos1, const vec3<double>& pos2, double& dist, vec3<double>& normal, double& coord, int index)
{
	/*if (!checkBB(pos1, pos2))
		return false;*/

	return false;
}

void LevelsetCollider::getGridDist(const nv::vec3<double>& pos, double& dist)
{
	const vec3<double> testPos = pos;

	const vec3<double> gridPos = (testPos - gridOrigin) * gridStepInv;
	if (gridPos[0] < 1. || gridPos[0] >= double(gridSize[0] - 2) ||
		gridPos[1] < 1. || gridPos[1] >= double(gridSize[1] - 2) ||
		gridPos[2] < 1. || gridPos[2] >= double(gridSize[2] - 2)){
		dist = DBL_MAX;
		return;
	}

	const vec3<int> gridPosI = vec3<int>(int(gridPos[0]), int(gridPos[1]), int(gridPos[2]));
	const int gridInd = dot(gridPosI,gridSizeOffset);

	const int gridInd1 = gridInd + 1;
	const vec3<double> gridPosF = vec3<double>(gridPos[0] - gridPosI[0], gridPos[1] - gridPosI[1], gridPos[2] - gridPosI[2]);
	const vec3<double> gridPosFInv = vec3<double>(1., 1., 1.) - gridPosF;
	const double zTerm0 =
		distGrid[gridInd] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm1 =
		distGrid[gridInd + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm2 =
		distGrid[gridInd1] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[2]] * gridPosF[2];
	const double zTerm3 =
		distGrid[gridInd1 + gridSizeOffset[1]] * gridPosFInv[2] +
		distGrid[gridInd1 + gridSizeOffset[1] + gridSizeOffset[2]] * gridPosF[2];
	const double yTerm0 =
		zTerm0 * gridPosFInv[1] +
		zTerm1 * gridPosF[1];
	const double yTerm1 =
		zTerm2 * gridPosFInv[1] +
		zTerm3 * gridPosF[1];
	dist =
		yTerm0 * gridPosFInv[0] +
		yTerm1 * gridPosF[0];
}
