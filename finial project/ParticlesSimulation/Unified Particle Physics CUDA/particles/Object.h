
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <PlyLoader.h>
#include <LevelsetCollider.h>
#include <fstream>
using namespace nv;
//using namespace std;
//extern "C"
//{
#ifndef OBJECT_H
#define OBJECT_H
#ifndef TRIPLE
#define TRIPLE
	typedef struct  triple
	{
		float x[3];
	} triple;
#endif
#ifndef __SDF__
#define __SDF__
	typedef struct SDF
	{
		float sdf;
		float gradient[3];
	}SDF;
#endif
	/*typedef enum ObjectType{ Particle, Rigid }ObjectType;
	typedef struct Object{

	ObjectType obj_type;
	uint num_particles;
	uint index_start;
	uint index_end;

	}MyObject;*/
#ifndef CMP_SDF
#define CMP_SDF
inline int cmp_sdf(const void* a, const void* b)
	{
		SDF *sdf_a, *sdf_b;
		sdf_a = (SDF *)a;
		sdf_b = (SDF *)b;
		if (sdf_a->sdf > sdf_b->sdf)
			return 1;
		else
			return -1;
	}
#endif
#ifndef __MAKECUBE__
#define __MAKECUBE__
inline 	void MakeCube(int length, int width, int height, double r, triple par_list[], SDF sdf_list[])
	{
		SDF sdf[6];
		int count = 0;
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int k = 0; k < height; k++)
				{
					par_list[count].x[0] = 2 * r*i;
					par_list[count].x[1] = 2 * r*j;
					par_list[count].x[2] = 2 * r*k;

					memset(sdf, 0, 6 * sizeof(SDF));

					sdf[0].sdf = (i - 0 + 0.5) * 2 * r;
					sdf[0].gradient[0] = -sdf[0].sdf;
					sdf[0].gradient[1] = 0;
					sdf[0].gradient[2] = 0;

					sdf[1].sdf = (j - 0 + 0.5) * 2 * r;
					sdf[1].gradient[0] = 0;
					sdf[1].gradient[1] = -sdf[0].sdf;
					sdf[1].gradient[2] = 0;

					sdf[2].sdf = (k - 0 + 0.5) * 2 * r;
					sdf[2].gradient[0] = 0;
					sdf[2].gradient[1] = 0;
					sdf[2].gradient[2] = -sdf[0].sdf;

					sdf[3].sdf = (length - 1 - i + 0.5) * 2 * r;
					sdf[3].gradient[0] = sdf[0].sdf;
					sdf[3].gradient[1] = 0;
					sdf[3].gradient[2] = 0;

					sdf[4].sdf = (width - 1 - j + 0.5) * 2 * r;
					sdf[4].gradient[0] = 0;
					sdf[4].gradient[1] = sdf[0].sdf;
					sdf[4].gradient[2] = 0;

					sdf[5].sdf = (height - 1 - k + 0.5) * 2 * r;
					sdf[5].gradient[0] = 0;
					sdf[5].gradient[1] = 0;
					sdf[5].gradient[2] = sdf[0].sdf;

					qsort(sdf, 6, sizeof(SDF), cmp_sdf);

					if (sdf[0].sdf != sdf[1].sdf)
					{
						memcpy(&sdf_list[count], &sdf[0], sizeof(SDF));
					}
					else
					{
						if (sdf[1].sdf != sdf[2].sdf)
						{
							sdf_list[count].sdf = (sdf[0].sdf + sdf[1].sdf) / 2;
							sdf_list[count].gradient[0] = (sdf[0].gradient[0] + sdf[1].gradient[0]) / 2;
							sdf_list[count].gradient[1] = (sdf[0].gradient[1] + sdf[1].gradient[1]) / 2;
							sdf_list[count].gradient[2] = (sdf[0].gradient[2] + sdf[1].gradient[2]) / 2;
						}
						else
						{
							sdf_list[count].sdf = (sdf[0].sdf + sdf[1].sdf + sdf[2].sdf) / 3;
							sdf_list[count].gradient[0] = (sdf[0].gradient[0] + sdf[1].gradient[0] + sdf[2].gradient[0]) / 3;
							sdf_list[count].gradient[1] = (sdf[0].gradient[1] + sdf[1].gradient[1] + sdf[2].gradient[1]) / 3;
							sdf_list[count].gradient[2] = (sdf[0].gradient[2] + sdf[1].gradient[2] + sdf[2].gradient[2]) / 3;
						}
					}
					float l_g = sqrt(sdf_list[count].gradient[0] * sdf_list[count].gradient[0] +
						sdf_list[count].gradient[1] * sdf_list[count].gradient[1] +
						sdf_list[count].gradient[2] * sdf_list[count].gradient[2]);
					sdf_list[count].gradient[0] /= l_g;
					sdf_list[count].gradient[1] /= l_g;
					sdf_list[count].gradient[2] /= l_g;
					sdf_list[count].sdf *= -1;

					count++;
				}
			}
		}
	}
#endif

inline int LoadFromPly(char filename[], int size, double r, triple par_list[], SDF sdf_list[])
{
	CPLYLoader plyLoader;
	plyLoader.LoadModel(filename);
	LevelsetCollider LC;
	LC.setMeshModel(plyLoader);
	LC.calculate(size, 0, 0);
	/*double df[10000];
	memcpy(df, LC.distGrid, LC.gridSize.x*LC.gridSize.x*LC.gridSize.z * sizeof(double));
	ofstream out;
	out.open("distGrid.txt");
	for (int i = 0; i < LC.gridSize.x*LC.gridSize.x*LC.gridSize.z; i++)
	{
		out << df[i] << endl;
	}*/
	float scale = r * 2 / LC.gridStep;
	vec3<double> center = vec3<double>(0,0,0);
	int index_par = 0;
	for (int i = 0; i < LC.gridSize.x; i++)
	{
		for (int j = 0; j < LC.gridSize.y; j++)
		{
			for (int k = 0; k < LC.gridSize.z; k++)
			{
				vec3<int> gridPos = vec3<int>(i, j, k);
				vec3<double> pos = vec3<double>(i + 0.5, j + 0.5, k + 0.5) * LC.gridStep + LC.gridOrigin;
				int gridInd = dot(gridPos, LC.gridSizeOffset);
				double sdf;
				LC.getGridDist(pos, sdf);
				if (sdf < 0 && sdf > -LC.gridStep*1.2)
				{
					//center += mass*pos * scale;
					par_list[index_par].x[0] = pos.x * scale;
					par_list[index_par].x[1] = pos.y * scale;
					par_list[index_par].x[2] = pos.z * scale;

					sdf_list[index_par].sdf = sdf * scale;
					vec3<double> gradient;
					LC.checkCollision(pos, sdf, gradient, -1);
					sdf_list[index_par].gradient[0] = gradient.x * scale;
					sdf_list[index_par].gradient[1] = gradient.y * scale;
					sdf_list[index_par].gradient[1] = gradient.z * scale;
					index_par++;
				}
			}
		}
	}
	return index_par;
}
//}
#endif