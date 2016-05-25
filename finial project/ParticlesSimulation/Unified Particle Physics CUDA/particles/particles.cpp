
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>
#include "Object.h"
// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>
#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"
#include "PlyLoader.h"
#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   16384

const uint width = 640, height = 480;
//CPLYLoader plyLoader;
// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::MESH_MODE;

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;

enum { M_VIEW = 0, M_MOVE };

uint numParticles = 0;
uint numObjects = 0;
uint3 gridSize;
int numIterations = 0; // run until exit
int Niteration = 0;
// simulation parameters
float timestep = 0.4f;//0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;

float collideSpring = 0.5f;;
float collideDamping = 0.02f;;
float collideShear = 0.1f;
float collideAttraction = 0.0f;
int k_mass_scale = 5;
float factor_s = 2.8;
float factor_k = 1.38;
float beta = 0.98;
float omega = 1.8;
bool friction = true;
bool mesh = false;
ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);






void Initialize()
{
	/*glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	plyLoader.LoadModel("bunny.ply");*/
}









// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL);

	psystem->reset(ParticleSystem::DEFORMATION_QUAD   );

    if (bUseOpenGL)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void runBenchmark(int iterations, char *exec_path)
{
    printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

    if (g_refFile)
    {
        printf("\nChecking result...\n\n");
        float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
        copyArrayFromDevice(hPos, psystem->getCudaPosVBO(),
                            0, sizeof(float)*4*psystem->getNumParticles());

        sdkDumpBin((void *)hPos, sizeof(float)*4*psystem->getNumParticles(), "particles.bin");

        if (!sdkCompareBin2BinFloat("particles.bin", g_refFile, sizeof(float)*4*psystem->getNumParticles(),
                                    MAX_EPSILON_ERROR, THRESHOLD, exec_path))
        {
            g_TotalErrors++;
        }
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
		char meshInf[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		std::string f;
		
		if (psystem->getFriction())
			sprintf(fps, "CUDA Particles (%d particles): %3.1f fps with friction", (int)psystem->getNumParticles(), ifps );
		else
			sprintf(fps, "CUDA Particles (%d particles): %3.1f fps without friction", (int)psystem->getNumParticles(), ifps);
		if (displayMode == ParticleRenderer::MESH_MODE)
		{
			sprintf(meshInf, " Face:%d Vertex:%d ", psystem->getNumFaces(), psystem->getNumVertexes());
			strcat(fps, meshInf);
		}

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void display()
{
    sdkStartTimer(&timer);

    // update the simulation
    if (!bPause)
    {
		//修改psystem中的参数
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);
		psystem->setK_Mass_Scale(k_mass_scale);
		psystem->setFactor_k(factor_k);
		psystem->setFactor_s(factor_s);
		psystem->setBeta(beta);
		psystem->setOmega(omega);
		if (displayMode == ParticleRenderer::MESH_MODE)
		{
			psystem->setMesh(true);
		}
		else
		{
			psystem->setMesh(false);
		}
		//psystem->setFriction(friction);
		numIterations++;
		/*if (numIterations % 100 == 0)
		{
			int a = 1;
			a = a + 1;
			printf("%d\n", a);
		}*/
        psystem->update(timestep);

        if (renderer)
        {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
			/*if (displayMode == ParticleRenderer::MESH_MODE)
			{
				
			}*/

        }
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
	GLfloat no_mat[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat mat_emission[] = { 0.3, 0.2, 0.2, 0.0 };
	glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
    glutWireCube(2.0);
	glMaterialfv(GL_FRONT, GL_EMISSION, no_mat);

    // collider
    glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(1.0, 0.0, 0.0);
    glutSolidSphere(psystem->getColliderRadius(), 20, 10);
    glPopMatrix();


	glEnable(GL_LIGHTING);

	//设置光源属性
	GLfloat light_ambient[] = { 1.0, 1.0, 1.0, 0.0 };
	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	//指定光源的位置
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 1.0 };

	//用定义好的光源属性给指定光源GL_LIGHT0进行设置
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	//开启设置的光源GL_LIGHT0
	glEnable(GL_LIGHT0);
    if (renderer && displayEnabled)
    {
		if (displayMode == ParticleRenderer::MESH_MODE)
		{
			
			psystem->renderMesh();
		}
		else
			renderer->display(displayMode);
		//plyLoader.Draw();
    }

    if (displaySliders)
    {
		//glColor3f(1, 0, 0);
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
		params->SetBarColorInner(1, 1, 1);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void addSphere()
{
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr+(pr*2.0f)*ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr*2.0f);
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer)
    {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders)
    {
        if (params->Mouse(x, y, button, state))
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
    r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
    r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
    r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
    r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode)
    {
        case M_VIEW:
            if (buttonState == 3)
            {
                // left+middle = zoom
                camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
            }
            else if (buttonState & 2)
            {
                // middle = translate
                camera_trans[0] += dx / 100.0f;
                camera_trans[1] -= dy / 100.0f;
            }
            else if (buttonState & 1)
            {
                // left = rotate
                camera_rot[0] += dy / 5.0f;
                camera_rot[1] += dx / 5.0f;
            }

            break;

        case M_MOVE:
            {
                float translateSpeed = 0.003f;
                float3 p = psystem->getColliderPos();

                if (buttonState==1)
                {
                    float v[3], r[3];
                    v[0] = dx*translateSpeed;
                    v[1] = -dy*translateSpeed;
                    v[2] = 0.0f;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }
                else if (buttonState==2)
                {
                    float v[3], r[3];
                    v[0] = 0.0f;
                    v[1] = 0.0f;
                    v[2] = dy*translateSpeed;
                    ixform(v, r, modelView);
                    p.x += r[0];
                    p.y += r[1];
                    p.z += r[2];
                }

                psystem->setColliderPos(p);
            }
            break;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(timestep);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
            }

            break;

        case '\033':
        case 'q':
            // cudaDeviceReset causes the driver to clean up all state. While
            // not mandatory in normal operation, it is good practice.  It is also
            // needed to ensure correct operation when the application is being
            // profiled. Calling cudaDeviceReset causes all profile data to be
            // flushed before the application exits
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
            break;

        case 'v':
            mode = M_VIEW;
            break;

        case 'm':
            mode = M_MOVE;
            break;

        case 'p':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
			if (displayMode == ParticleRenderer::MESH_MODE)
			{
				psystem->setMesh(true);
			}
			else
			{
				psystem->setMesh(false);
			}
            break;

        case 'd':
            psystem->dumpGrid();
            break;

        case 'u':
            psystem->dumpParticles(0, numParticles-1);
            break;
		case 'f':
			psystem->setFriction(!psystem->getFriction());
			break;
        case 'r':
            displayEnabled = !displayEnabled;
            break;

        case '1':
            psystem->reset(ParticleSystem::COLLISION_DEMO);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
            break;

        case '2':
            psystem->reset(ParticleSystem::DEFORMATION_LINEAR);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
            break;

        case '3':
			psystem->reset(ParticleSystem::DEFORMATION_QUAD);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
            break;

        case '4':
			psystem->reset(ParticleSystem::GRANULAR_FRICTION);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
            break;
		case '5':
			psystem->reset(ParticleSystem::MESH_DEMO);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
			break;
		case '6':
			psystem->reset(ParticleSystem::STACK_DEMO);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
			break;
		case '7':
			psystem->reset(ParticleSystem::BUNNY_PILE);
			renderer->setParticleRadius(psystem->getParticleRadius());
			renderer->setColorBuffer(psystem->getColorBuffer());
			break;
        case 'w':
            wireframe = !wireframe;
            break;

        case 'h':
            displaySliders = !displaySliders;
            break;
    }

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders)
    {
        params->Special(k, x, y);
    }

    demoMode = false;
    idleCounter = 0;
}

void idle(void)
{
    if ((idleCounter++ > idleDelay) && (demoMode==false))
    {
        //demoMode = true;
        //printf("Entering demo mode\n");
    }

    if (demoMode)
    {
        camera_rot[1] += 0.1f;

        if (demoCounter++ > 1000)
        {
            ballr = 10 + (rand() % 10);
            addSphere();
            demoCounter = 0;
        }
    }

    glutPostRedisplay();
}

void initParams()
{
    if (g_refFile)
    {
        timestep = 0.0f;
        damping = 0.0f;
        gravity = 0.0f;
        ballr = 1;
        collideSpring = 0.0f;
        collideDamping = 0.0f;
        collideShear = 0.0f;
        collideAttraction = 0.0f;

    }
    else
    {

        // create a new parameter list
        params = new ParamListGL("misc");
        params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.018f, &timestep));
        params->AddParam(new Param<float>("damping"  , damping , 0.0f, 1.0f, 0.001f, &damping));
        params->AddParam(new Param<float>("gravity"  , gravity , 0.0f, 0.001f, 0.0003f, &gravity));
		params->AddParam(new Param<int>("k", k_mass_scale, 0, 5, 5, &k_mass_scale));
		params->AddParam(new Param<float>("mu_s", factor_s, 0.0f, 5.0f, 2.8f, &factor_s));
		params->AddParam(new Param<float>("mu_k", factor_k, 0.0f, 5.0f, 1.38f, &factor_k));
		params->AddParam(new Param<float>("beta", beta, 0.0f, 1.0f, 0.88f, &beta));
		params->AddParam(new Param<float>("omega", omega, 1.0f, 2.0f, 1.5, &omega));
        /*params->AddParam(new Param<int> ("ball radius", ballr , 1, 20, 1, &ballr));

        params->AddParam(new Param<float>("collide spring" , collideSpring , 0.0f, 1.0f, 0.001f, &collideSpring));
        params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
        params->AddParam(new Param<float>("collide shear"  , collideShear  , 0.0f, 0.1f, 0.001f, &collideShear));
        params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));*/
    }
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset as Rigid Collision Demo [1]", '1');
    glutAddMenuEntry("Reset as Linear Defomation [2]", '2');
    glutAddMenuEntry("Rest as  Quadratic Defromation [3]", '3');
	glutAddMenuEntry("Rest as  Granular Friction [4]", '4');
	glutAddMenuEntry("Rest as  Bunny Demo [5]", '5');
	glutAddMenuEntry("Rest as  Stack Demo [6]", '6');
	glutAddMenuEntry("Rest as  Bunny Pile [7]", '7');
	//glutAddMenuEntry("Rest as  Granular Friction [4]", '4');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Toggle friction [f]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
 {
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif
	
    printf("%s Starting...\n\n", sSDKsample);

	numParticles = 0; //NUM_PARTICLES;
	numObjects = 0;
    uint gridDim = GRID_SIZE;
    numIterations = 0;

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
        {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "grid"))
        {
            gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
            fpsLimit = frameCheckNumber;
            numIterations = 1;
        }
    }

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    //printf("particles: %d\n", numParticles);

    bool benchmark = checkCmdLineFlag(argc, (const char **) argv, "benchmark") != 0;

    if (checkCmdLineFlag(argc, (const char **) argv, "i"))
    {
        numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
    }

    if (g_refFile)
    {
        cudaInit(argc, argv);
    }
    else
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }

        initGL(&argc, argv);
        cudaGLInit(argc, argv);
    }

    initParticleSystem(numParticles, gridSize, g_refFile==NULL);
    initParams();
	//Initialize();
    if (!g_refFile)
    {
        initMenus();
    }

    if (benchmark || g_refFile)
    {
        if (numIterations <= 0)
        {
            numIterations = 300;
        }

        runBenchmark(numIterations, argv[0]);
    }
    else
    {
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(key);
        glutSpecialFunc(special);
        glutIdleFunc(idle);

        glutCloseFunc(cleanup);

        glutMainLoop();
    }

    if (psystem)
    {
        delete psystem;
    }
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

