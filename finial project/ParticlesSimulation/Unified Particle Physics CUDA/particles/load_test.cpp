//#include "PLYLoader.h"
//int width = 640;
//int height = 480;
//
//CPLYLoader plyLoader;
//
//void Initialize()
//{
//	glClearColor(0.0, 0.0, 0.0, 0.0);
//	glEnable(GL_DEPTH_TEST);
//	plyLoader.LoadModel("bunny.ply");
//}
//
//
//void Reshape(int w, int h)
//{
//	width = w;
//	height = h;
//	glViewport(0, 0, width, height);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluPerspective(45.0, (double)width / (double)height, 1.0, 1000.0);
//}
//
//void Display()
//{
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glViewport(0, 0, width, height);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//	glTranslatef(0.7, -0.3, 0);
//	glRotatef(280, 1, 0, 0);
//	glRotatef(80, 0, 0, 1);
//	glScalef(5, 5, 5);
//	plyLoader.Draw();
//	glutSwapBuffers();
//}
//
//int main(int argc, char **argv)
//{
//	glutInit(&argc, argv);
//	glutInitWindowPosition(0, 0);
//	glutInitWindowSize(width, height);
//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
//	glutCreateWindow("Load Model");
//	glutDisplayFunc(Display);
//	glutReshapeFunc(Reshape);
//
//	Initialize();
//
//	glutMainLoop();
//
//	return 0;
//}