from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import math
k = 2

def loadData(filename, f= 3):
	fin=open(filename,"r")
	data=array([])
	for line in fin:
		#print 1
		strlist =line.split(',')
		if len(strlist)== 65:
			#print 2
			x=zeros(64)
			for i in range(64):
				x[i]=float(strlist[i])
			c = int(strlist[64]);

			if(c == f):
				#print c
				if len(data)==0:
					data=x
				else:
					data=vstack((data,x))
	fin.close()
	#print data
	return data
def SortW(v,w):
	num = v.size
	#index = num - 1 - array(v.argsort())
	index = array(v.argsort())
	#print v
	#print index
	w_sorted = zeros(num*num).reshape(num,num)
	for i in range(index.size):
		for j in range(w.shape[0]):
			w_sorted[i][j] = w[j,index[num - i - 1]]
	return w_sorted

def train(data):
	u=data.sum(axis=0)/data.shape[0]
	data=data-u
	#v,w=linalg.eig(mat(data).T*mat(data)/(data.shape[0]-1))
	v,w=linalg.eig(mat(data).T*mat(data))#/(data.shape[0]-1))
	#w=w.T
	w_sorted = SortW(v,w)
	return w_sorted,u
def PCA_Test(data_test, e, u):
	data = data_test - u;
	w = mat(data)*mat(e).T
	#w = mat(data)*(mat(e)*mat(e).T)
	#w = (mat(e).T*mat(e))*mat(data).T
	#w = w.T
	#print u
	num = data.shape[0]
	print w[0]
	data_pca = array([])
	for i in range(num):
		x_pca = zeros(data.shape[1])
		for j in range(k):
			x_pca += array(w[i,j]).squeeze()*e[j]
		x_pca += u
		if len(data_pca)==0:
			data_pca=x_pca
		else:
			data_pca=vstack((data_pca,x_pca))
	return data_pca

def stitch(data, n_c):
	n_c = int(n_c)
	n_r = int(math.ceil(data.shape[0]/n_c))
	data_bp = zeros(8*8*n_r*n_c).reshape(8*n_r, 8*n_c)
	#print data.shape[0]
	for i in range(n_r):
		for j in range(n_c):
			#print i,j
			index_picture = i*n_c + j
			if index_picture >= data.shape[0]:
				break
			r_start = i*8
			c_start = j*8
			for k in range(64):
				
				#print index_picture,index_pixel
				r = k/8
				c = k - 8*r
				index_pixel = i*8 + c
				#print index_picture,index_pixel
				r = k/8
				r += r_start
				c += c_start
				data_bp[r,c] = data[index_picture][k]
	return data_bp

def PCA(trainname,testname):
	data_train = loadData(trainname,3)
	data_test = loadData(testname,3)
	e,u = train(data_train)
	data_pca = PCA_Test(data_test, e, u)
	bp_pca = stitch(data_pca,15)
	bp_test = stitch(data_test,15)
	fig_test = plt.figure() 
	fig_pca = plt.figure()
	ax_test = fig_test.add_subplot(111)
	ax_test.set_title("test data")
	ax_test.imshow(bp_test, cmap='gray', vmin = 0, vmax = 16)
	ax_pca = fig_pca.add_subplot(111)
	ax_pca.set_title('PCA')
	ax_pca.imshow(bp_pca, cmap='gray', vmin = 0, vmax = 16)
	plt.show()



PCA('optdigits.tra', 'optdigits.tes')
