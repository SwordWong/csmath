from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

e=0.00000001
ie=1.0/e

def funcxy(X,Y):
	return X*X+Y*Y
def func(X):
	return funcxy(X[0],X[1])
def fderiv(X,i):
	if i==0:
		return 2.0*X[0]
	elif i==1:
		return X[1]
	else:
		return 0.0
def fderiv2(X,i,j):
	if i==0 and j==0:
		return 2.0
	elif i==0 and j==1:
		return 0.0
	elif i==1 and j==0:
		return 0.0
	elif i==1 and j==1:
		return 1.0
	else:
		return 0.0

def positive_definite(M):
	w,v=linalg.eig(M)
	t=w.min()
	if t>=0: return 1
	else: return 0
def calg(X):
	n=X.size
	g=zeros(n)
	g.shape=(n,1)
	for i in range(X.size):
		g[i]=fderiv(X,i)
	return g
def calG(X):
	n=X.size
	G=mat(arange(0.0, 1.0*n*n,1))
	G.shape=(n,n)
	for i in range(n):
		for j in range(n):
			G[i,j]=fderiv2(X,i,j)
	return G
def NP_iter(xi,yi,u):
	x=array([xi,yi])
	x=x.reshape(x.size,1)	
	f=func(x)
	g=calg(x)
	G=calG(x)
	X=array([xi]).reshape(1,1)
	Y=array([yi]).reshape(1,1)
	Z=array([f]).reshape(1,1)
	while vdot(g,g)>=e*e:
		G2=G.copy()
		while positive_definite(G2+identity(x.size)*u)==0:
			u*=4
		G2+=identity(x.size)*u
	
		A=mat(G2)
		B=mat(-g)
		s=linalg.solve(A,B)
	
		f2=func(x+s)
		df=f2-f
		dq=dot(g.T,s)+0.5*dot(dot(s.T,G),s)
		rk=dq/df
		if rk<0.25:
			u*=4
		elif rk>0.75:
			u*=0.5
		if rk>0.0:
			xi2=(x[0]+s[0])
			yi2=(x[1]+s[1])
			x2=array([xi2,yi2])
			x2=x2.reshape(x2.size,1)
			#
			X=concatenate((X,xi2))
			Y=concatenate((Y,yi2))
			Z=concatenate((Z,f2.reshape(1,1)))
			#
			x=x2
			f=f2
		g=calg(x)
		G=calG(x)
	return X,Y,Z
def init(width = 15.0, num = 50):
	#m=15.0
	#n=50
	ud=2*width/num
	xx=arange(-width,width,ud)+ud*0.5
	yy=arange(-width,width,ud)+ud*0.5
	xx=xx.reshape(num,1)
	yy=yy.reshape(1,num)
	X=xx
	Y=yy
	for i in range(num-1):
		X=hstack((X,xx))
		Y=vstack((Y,yy))
	Z=funcxy(X,Y)
	return X,Y,Z

def drawIterationProcess(ax,X,Y,Z):
	X=array(X).squeeze()
	Y=array(Y).squeeze()
	Z=array(Z).squeeze()
	ax.plot3D(X, Y, Z, 'bo-',label='Iteration process')
	ax.legend()
	plt.show()
	
#def draw(X,Y,Z):
def NP(width,num, x_start,y_start):
	X,Y,Z = init(width,num)
	fig = plt.figure(figsize = (12, 8))
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0,alpha=0.4)
	cset = ax.contour(X, Y, Z, zdir='z', offset=-50)
	cset = ax.contour(X, Y, Z, zdir='x', offset=-width)
	cset = ax.contour(X, Y, Z, zdir='y', offset=width)
	X,Y,Z =  NP_iter(x_start,y_start,1)
	ax.set_xlabel('X')
	ax.set_xlim(-width, width)
	ax.set_ylabel('Y')
	ax.set_ylim(-width, width)
	ax.set_zlabel('Z')
	drawIterationProcess(ax,X,Y,Z)


NP(15.0,50,15.0,10.0)
	
	
	
	
	
