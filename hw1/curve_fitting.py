import numpy 
import matplotlib
import matplotlib.pyplot as plt
from math import exp
def Curve(x):
	curve = numpy.sin(2.0*numpy.pi*x)
	return curve
	
def getTargetCurve(x_min = 0.0, x_max = 1.0, N = 2048):
	u=(x_max-x_min)/N
	x = numpy.arange(x_min,x_max,u)+u*0.5
	curve = Curve(x)
	data = numpy.vstack((x,curve));
	return data
	
def getSampleData(x_min = 0.0, x_max = 1.0, N = 10, jitter = 0.3):
	u=(x_max-x_min)/N
	x = numpy.arange(x_min,x_max,u)+u*0.5
	#x += numpy.random.uniform(0,u,N) - u/2
	curve = Curve(x)
	curve += numpy.random.uniform(0.0,jitter,N) - jitter/2
	data = numpy.vstack((x,curve));
	return data
	
def getPolynomialFitData(data_sample, M, lamda = 0.0, N = 2048):
	#print x_max,x_min
	num = data_sample[0].size
	x = data_sample[0].reshape(num,1)
	y = data_sample[1].reshape(num,1)
	X = numpy.ones((num,1))
	for i in range(M):
		tmp = x**(i+1)
		X	= numpy.hstack((X,tmp))
	print x
	print X
	X = numpy.mat(X.copy())
	Y = numpy.mat(y.copy())
	#Least squares
	A=X.T * X + lamda*numpy.identity(M + 1)
	B=X.T * Y
	#W=numpy.linalg.solve(A,B)
	W=A.I*B
	print W
	x_max = x[x.size - 1]
	x_min = x[0]
	u=(x_max-x_min)/ N
	#print x_max,x_min,u
	x = numpy.arange(x_min,x_max,u)+u*0.5
	X = numpy.ones((x.size,1))
	for i in range(M):
		tmp = x**(i+1)
		X	= numpy.hstack((X,tmp.reshape(x.size,1)))
	X = numpy.mat(X.copy())
	Y = X*W
	y = numpy.array(Y).reshape(x.size,)
	data = numpy.vstack((x,y))
	return data
		
	
def myProgram(x_min = 0.0, x_max = 1.0, N = 10, M = 3, lamda = 0):
	data_target = getTargetCurve(x_min,x_max)
	data_sample = getSampleData(x_min, x_max, N)
	data_fit = getPolynomialFitData(data_sample, M, lamda,)
	
	fig, ax = plt.subplots()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('polynomial fit '+' M=' + str(M) + " N=" + str(N) + " lamda=" + str(lamda))
	ax.plot(data_target[0], data_target[1], '-g')
	ax.plot(data_sample[0], data_sample[1], '.b')
	ax.plot(data_fit[0], data_fit[1], '-r')
	plt.show()
	
myProgram(0.0, 1.0, 10, 9, exp(-18))