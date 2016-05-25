#pragma once

//#include "MeshModel.h"
//#include "Collider.h"
//
//#include "Utils/Container.h"
//#include "Utils/Transform.h"
//
//#include <QString>
#ifndef HELPER_MATH_H
#include <nvVector.h>
#endif
#include <PlyLoader.h>
class LevelsetCollider
{
public:

	LevelsetCollider								();
	~LevelsetCollider								();

	/*bool				load						(QString fileName, double offset);
	bool				save						(QString fileName);*/

	void				setMeshModel(CPLYLoader& model);
	void				calculate					(int size, double offset, double margin);

	bool				testCollision				(const nv::vec3<double>& pos, int index = -1);
	bool				testCollision				(const nv::vec3<double>& pos1, const nv::vec3<double>& pos2, int index = -1);

	bool				checkCollision				(const nv::vec3<double>& pos, double& dist, nv::vec3<double>& normal, int index = -1);
	bool				checkCollision				(const nv::vec3<double>& pos1, const nv::vec3<double>& pos2, double& dist, nv::vec3<double>& normal, double& coord, int index = -1);

	void				getGridDist					(const nv::vec3<double>& pos, double& dist);

	CPLYLoader*			meshModel;

	nv::vec3<double>	gridOrigin;
	nv::vec3<int>		gridSize;
	nv::vec3<int>		gridSizeOffset;

	nv::vec3<double>	MaxBoundary;
	nv::vec3<double>	MinBoundary;

	double				gridStep;
	double				gridStepInv;

	double				testThreshold;

	double				gridOffset;
	double				gridMargin;

	double*				distGrid;
};
