from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.animation as anim

#
# preset gaussian distributed in (-1,1)
# per gaussian distributes 0.3 area
#
center_amount=8
point_amount=1000
wj=zeros(0)
def onepass(fig,ax,sx,sy,sc,gaussians,cx,cy,scalarMap):
		
		# E-pass
		for i in range(point_amount):
			ilike=0.0
			for j in range(center_amount):
				wj[j]=gaussians[j].P(sx[i],sy[i])
				ilike+=wj[j]*gaussians[j].prior
			maxw=0.0
			maxj=-1
			for j in range(center_amount):
				w=wj[j]*gaussians[j].prior/ilike
				wij[i,j]=w
				if w>maxw:
					maxw=w
					maxj=j
			sc[i]=maxj
		# M-pass
		pj=wij.sum(axis=0)
		psum=pj.sum()
		assert(pj.size==center_amount)
		for j in range(center_amount):
			gaussians[j].prior=pj[j]/point_amount
			sxm=(sx*wij[:,j]).sum()/pj[j]
			sym=(sy*wij[:,j]).sum()/pj[j]
			mean0=gaussians[j].mean
			gaussians[j].mean=array([sxm,sym])
			sx2=sx.copy()-sxm
			sy2=sy.copy()-sym
			var=zeros(4).reshape(2,2)
			for i in range(point_amount):
				sxy=mat(array([sx2[i],sy2[i]]))
				cov=sxy.T*sxy
				var+=wij[i,j]*cov
			var/=pj[j]
			gaussians[j].variance=var
class Gaussian(object):
	dim = 2
	prior = 0.0
	mean = zeros(2)
	variance = identity(2)*0.1
	def P(self,xi,yi):
		x=array([xi,yi])
		u=self.mean
		s=self.variance
		det=linalg.det(s)
		xu=mat(x-u).T
		si=mat(s).I
		p=exp(-0.5*xu.T*si*xu)/(sqrt(pow((2.0*pi),self.dim)) * sqrt(det))
		return p
def getCenter(n_c):
	cx=random.uniform(-1.0,1.0,n_c)
 	cy=random.uniform(-1.0,1.0,n_c)
 	return cx,cy
def getSamples(n_c,n_p):
	per_center=n_p/n_c
	sx=random.randn(per_center)*0.2+cx[0]
	sy=random.randn(per_center)*0.2+cy[0]
	for i in range(n_c-1):
		sx=concatenate((sx,random.randn(per_center)*0.3+cx[i+1]))
		sy=concatenate((sy,random.randn(per_center)*0.3+cy[i+1]))
	sc=floor(random.uniform(0,n_c-1,n_p))
	return sx,sy,sc
def draw(sx,sy,sc,cx,cy,gaussians):
	fig.clf()
	center_amount = len(cx)
	for i in range(sx.size):
		c=sc[i]
		colorVal = scalarMap.to_rgba(c)
		plt.plot(sx[i],sy[i],'o',markersize=3,color=colorVal)
	for i in range(cx.size):
		colorVal = scalarMap.to_rgba(i)
		plt.plot(cx[i],cy[i],'o',markersize=9,color=colorVal)
	plt.show()

if __name__=="__main__":
	fig=plt.figure(figsize = (8, 8))
	ax=fig.add_subplot(111)
	jet = plt.get_cmap('jet')
	cNorm  = colors.Normalize(vmin=0, vmax=center_amount-1)
	scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
 	cx,cy = getCenter(center_amount)
	sx,sy,sc = getSamples(center_amount, point_amount)
	gaussians=[]
	for i in range(center_amount):
		gaussian=Gaussian()
		gaussian.prior=1.0/center_amount
		gaussian.mean=array([cx[i],cy[i]])
		gaussians.append(gaussian)
	wij=zeros(center_amount*point_amount).reshape(point_amount,center_amount)
	wj=zeros(center_amount)
	for i in range(5):
		onepass(fig,ax,sx,sy,sc,gaussians,cx,cy,scalarMap)
	draw(sx,sy,sc,cx,cy,gaussians)

#EM()
	

	

