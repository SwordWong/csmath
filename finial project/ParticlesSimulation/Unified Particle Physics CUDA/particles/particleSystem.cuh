
#include "particles_kernel.cuh"
//#ifndef PSCUDA
//#define PSCUDA
extern "C"
{
#ifndef OBJECTTYPE
#define OBJECTTYPE
	typedef enum ObjectType{ Particle, Rigid, Deformation_linear, Deformation_linear_plasticity,Deformation_quad, Deformation_quad_plasticity }ObjectType;
#endif
#ifndef ITERATIONTYPE
#define ITERATIONTYPE
	typedef enum IterationType{ Pre_Stablization, Main_Constaint }IterationType;
#endif
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
                         float deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells);
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
		float *olddVel,
		float *oldRelative,
		float *oldSDF,
		float *oldSDFgradient,
		int *oldPhase,
		float *oldInvMass,
		uint   numParticles,
		uint   numCells);
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
		);

    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
	float4 Cal_Center(float* dPos_exp, float* dInvMass, uint Start, uint Size, float dObjecMass);
	void Cal_Center_BR(float* dPos_exp, float* dCenter, float* dInvMass, uint* m_dObjectStart, uint* m_dObjectSize, ObjectType* dObjectType, float* m_dObjectMass, int num_object);
	void my_integrateSystem(float *pos,
		float *pos_exp,
		float *vel,
		float deltaTime,
		uint numParticles);
	void Cal_Matrix_A(int size, float4 center, float* dPos_exp, float* dRelative, float* result);
	void Cal_Matrix_A_BR(float* dPos_exp, float* dCenter, int* dPhase, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType, float* result, int num_Particle, int num_Object);
	void Cal_Matrix_A_quad(int size, float4 center, float* dPos_exp, float* dRelative, float* result);
	void Cal_Matrix_A_quad_BR(float* dPos_exp, float* dCenter, int* dPhase, uint* dObjectStart, uint* dObjectSize, float* dRelative, ObjectType* dObjectType, float* result, int num_Particle, int num_Object);

	//void Shaping_Cal_pos_kernel(float *dPos_exp, float* dRelative, int *dPhase, float* dCenter, float *dRoate, ObjectType *dObjectType, uint num_Particles);
	//void Shaping_Cal_pos_kernel(float *dPos_exp, float* dRelative, int *dPhase, float* dCenter, float *dRoate, float* dRoate_quad, ObjectType *dObjectType, uint num_Particles);
	void Shaping_Upadate_Pos_Exp(float *dPos_exp, float* dRelative, int *dPhase, float* dCenter, float *dRoate, float* dRoate_quad, ObjectType *dObjectType, uint num_Particles);
	void Update_Change(float *dPos, float *dPos_exp, float *dVel, int numParticles, float delta_time, float epsilon);
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
		);
	void Update_dRelative(float* dSp, float* dRelavite, float* A_qq, float* A_qq_quad, int num);
	void Update_Mesh(float* dVertex, float* dNormal, uint* dFace, float *dRelative, float* dRoate, float* dRoate_quad, float* dcenter, int num_face, int num_vertex, ObjectType type);
	void Update_MeshFullPal(float* dVertex, float* dNormal, uint* dFace, float* Normal_Original, ObjectType *dObjectType, float *dRelative, int * dPhaseVertex, int * dPhaseFace, float* dRoate, float* dRoate_quad, float* dcenter, int num_face, int num_vertex);
	void Boundary(float3 MaxB, float3 MinB, float4* oldPos, float4* dPos_exp, float r, int num_particle, IterationType itType);
}
//#endif