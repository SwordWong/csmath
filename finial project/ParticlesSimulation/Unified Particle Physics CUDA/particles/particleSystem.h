

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <ParticleMesh.h>
#include <helper_functions.h>
#include <fstream>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "Object.h"

#ifndef TRIPLE
#define TRIPLE
typedef struct  triple
{
	float x[3];
} triple;
#endif
#define MAX_PARTICLES  30000
#define MAX_OBJECT 120
#define MAX_VERTEX MAX_OBJECT * 25000
#define MAX_FACE MAX_OBJECT * 45000
#ifndef OBJECTTYPE
#define OBJECTTYPE
typedef enum ObjectType{ Particle, Rigid, Deformation_linear, Deformation_linear_plasticity, Deformation_quad, Deformation_quad_plasticity }ObjectType;
#endif
#ifndef ITERATIONTYPE
#define ITERATIONTYPE
typedef enum IterationType{ Pre_Stablization, Main_Constaint }IterationType;
#endif
// Particle system class
class ParticleSystem
{
    public:
		//typedef enum ObjectType{ Particle, Rigid }ObjectType;
        ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS,
			NO_OPERATION,
			ADD_RIGID,
			COLLISION_DEMO,
			STACK_DEMO,
			GRANULAR_FRICTION,
			DEFORMATION_LINEAR,
			DEFORMATION_QUAD,
			DEFORMATION_LINEAR_PLA,
			DEFORMATION_QUAD_PLA,
			MESH_DEMO,
			BUNNY_PILE
        };

        enum ParticleArray
        {
			EMPTY,
			POSITION,
			POSITION_CONTACT,
			VELOCITY,
			INVERSE_MASS,
			PHASE,
			SDF_arr,
			SDF_GRADIENT,
			RELATIVE_POS,
			FORCE,
			ROATE,
			ROATE_QUAD,
			OBJECT_SIZE,
			OBJECT_START,
			OBJECT_END,
			OBJECT_TYPE,
			OBJECT_MASS,
			OBJECT_CENTER,
			OBJECT_A_QQ,
			OBJECT_A_QQ_QUAD,
			OBJECT_SP,
			VERTEX,
			FACE,
			VERTEX_COLOR,
			NORMAL,
			NORMAL_ORIGINAL,
			OBJECT_VERTEX_START,
			OBJECT_FACE_START,
			RELATIVE_VERTEX,
			PHASE_VERTEX,
			PHASE_FACE,
			OBJECT_NUM_VERTEX,
			OBJECT_NUM_FACE

        };

        void update(float deltaTime);
        void reset(ParticleConfig config);

        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const void *data, int start, int count);
		void addRigid(triple postion_start, triple  par_list[], SDF sdf_list[],int n_par_in_rig, float m, float r);
		void addParticleGroup(triple postion_start, triple  par_list[], int n_par_in_group, float m, float r);
		void addDeformation(triple postion_start, triple  par_list[], SDF sdf_list[], int n_par_in_rig, float m, float r, ObjectType type);
		void addMesh(int index_obj, ParticleMesh &PM);
		void addMesh(int index_obj, ParticleMesh &PM, float3 color);
		void Upate_dRelative(float* dSp, float* dRelavite, float* A_qq, float* A_qq_quad, int num);
        int    getNumParticles() const
        {
            return m_numParticles;
        }
		int    getNumFaces() const
		{
			return m_numFaces;
		}
		int    getNumVertexes() const
		{
			return m_numVertexes;
		}
        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }
		/*unsigned int getVertexBuffer(int index_obj)       const
		{
			return m_hPM[index_obj].m_vertexVbo;
		}
		unsigned int getVIndexBuffer(int index_obj)       const
		{
			return m_hPM[index_obj].m_indexVBO;
		}*/

        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }
        void *getCudaColorVBO()            const
        {
            return (void *)m_cudaColorVBO;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count);

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
		void setK_Mass_Scale(int k)
		{
			m_params.k_mass_scale = k;
		}
		void setFactor_s(float factor_s)
		{
			m_params.factor_s = factor_s;
		}
		void setBeta(float beta)
		{
			 m_params.beta = beta;
		}
		void setFactor_k(float factor_k)
		{
			m_params.factor_k = factor_k;
		}
		void setOmega(float Omega)
		{
			m_params.Omega = Omega;
		}
		void setFriction(bool friction)
		{
			m_params.friction = friction;
		}
		void setMesh(bool mesh)
		{
			m_params.mesh = mesh;
		}
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }

        void setColliderPos(float3 x)
        {
            m_params.colliderPos = x;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        float3 getColliderPos()
        {
            return m_params.colliderPos;
        }
        float getColliderRadius()
        {
            return m_params.colliderRadius;
        }
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }
		bool getFriction()
		{
			return m_params.friction;
		}
		uint getnumObject()
		{
			return m_numObjects;
		}
		/*ParticleMesh* getPM()
		{
			return m_hPM;
		}*/
		void shaping(float deltaTime);
		void shaping_BR(float deltaTime);
		void shaping_with_cub();
        void addSphere(int index, float *pos, float *vel, int r, float spacing);
		void renderMesh();
		std::fstream out;



		






    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);

    protected: // data
        bool m_bInitialized, m_bUseOpenGL;
        uint m_numParticles;
		uint m_numObjects;
		uint m_numVertexes;
		uint m_numFaces;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities
		float *m_hInvm;			   //particle inverse mass
		int *m_hPhase;				//particle phase
		float* m_hSDF;
		float* m_hSDF_gradient;
		float* m_hRelative;
		float* m_hForce;
		
		float* m_hColorVertex;

		int* m_hPhaseVertex;
		int* m_hPhaseFace;

		int* m_hVertexStart;
		int* m_hFaceStart;

		uint* m_hObjectNumVertex;
		uint* m_hObjectNumFace;

		//float m_hPos[4*MAX_PARTICLES];              // particle positions
		//float m_hVel[4 * MAX_PARTICLES];              // particle velocities
		//float m_hInvm[MAX_PARTICLES];			   //particle inverse mass
		//int m_hPhase[MAX_PARTICLES];				//particle phase
		//float m_hSDF[MAX_PARTICLES];
		//float m_hSDF_gradient[4 * MAX_PARTICLES];


        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

		uint  *m_hObjectSize;
		uint  *m_hObjectStart;
		uint  *m_hObjectEnd;
		float *m_hObjectMass;
		float *m_hObjectSp;
		ObjectType *m_hObjectType;
		float* m_hRoate;
		float* m_hRoate_quad;
		float* m_hA_qq;
		float* m_hA_qq_quad;
		//ParticleMesh *m_hPM;
        
		// GPU data
        float *m_dPos;
		float *m_dPos_exp;
		//float *m_dPos_contact;
        float *m_dVel;
		float *m_dInvm;
		int *m_dPhase;
		float *m_dSDF;
		float* m_dSDF_gradient;
		float* m_dRelative;
		
		float* m_dForce;

		float* m_dVertex;
		float* m_dNormal;
		float* m_dFace;
		float* m_dRelativeVertex;
		float* m_dColorVertex;

		int* m_dPhaseVertex;
		int* m_dPhaseFace;

		int* m_dVertexStart;
		int* m_dFaceStart;

		uint* m_dObjectNumVertex;
		uint* m_dObjectNumFace;


        float *m_dSortedPos;
        float *m_dSortedVel;
		float *m_dSortedPos_exp;
		float *m_dSortedRelative;
		float *m_dSortedSDF;
		float *m_dSortedSDF_gradient;
		int	  *m_dSortedPhase;
		float *m_dSortedInvMass;
		ObjectType *m_dSortedObectType;

		long long num_iter;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell
		

		uint  *m_dObjectSize;
		uint  *m_dObjectStart;
		uint  *m_dObjectEnd;
		float *m_dObjectMass;
		ObjectType  *m_dObjectType;
		float* m_dRoate;
		float* m_dA_qq;
		float* m_dA_qq_quad;
		float* m_dRoate_quad;
		float* m_dSp;
		float* m_dCenter;
		

        uint   m_gridSortBits;

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint   m_colorVBO;          // vertex buffer object for colors

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

		uint   m_vertexVbo;            // vertex buffer object for particle positions
		uint   m_indexVBO;
		uint   m_vertexcolorVBO;
		uint   m_NorVBO;
		uint   m_NorOriginalVBO;

		float *m_cudaVerVBO;        // these are the CUDA deviceMem Pos
		float *m_cudaIndexVBO;      // these are the CUDA deviceMem Color


        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange


		struct cudaGraphicsResource *m_cuda_vertexvbo_resource; // handles OpenGL-CUDA exchange
		struct cudaGraphicsResource *m_cuda_indexvbo_resource; // handles OpenGL-CUDA exchange
		struct cudaGraphicsResource *m_cuda_vertexcolorvbo_resource; // handles OpenGL-CUDA exchange
		struct cudaGraphicsResource *m_cuda_normalvbo_resource; // handles OpenGL-CUDA exchange
		struct cudaGraphicsResource *m_cuda_normal_original_vbo_resource;

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;







};

#endif // __PARTICLESYSTEM_H__
