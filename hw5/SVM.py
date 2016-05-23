import numpy as np
import matplotlib.pyplot as plt

class SVM(object):

	nonlinear = 0

	N = 0
	d = 0
	C = 0.05
	tolerance = 0.001
	eps = 0.001
	twoSigma = 2.

	b = 0.
	alph = None
	w = None

	errorCache = None

	densePoints = None
	target = None
	precomputedDot = None
	precomputedSelfDot = None

	firstTestI = 0
	endSupportI = 0

	def examineExample(self, i1):
		y1 = self.target[i1]
		alph1 = self.alph[i1]
		E1 = 0.
		if alph1 > 0. and alph1 < self.C:
			E1 = self.errorCache[i1]
		else:
			E1 = self.learnedFunc(i1) - y1
		r1 = y1 * E1
		if (r1 < -self.tolerance and alph1 < self.C) or (r1 > self.tolerance and alph1 > 0.):
			i2 = -1
			tmax = 0.
			for k in range(0, self.endSupportI):
				if self.alph[k] > 0. and self.alph[k] < self.C:
					E2 = self.errorCache[k]
					temp = abs(E1 - E2)
					if temp > tmax:
						tmax = temp
						i2 = k
			if i2 >= 0:
				if self.takeStep(i1, i2) == 1:
					return 1
			k0 = np.random.randint(0, self.endSupportI)
			for k in range(k0, k0 + self.endSupportI):
				i2 = k % self.endSupportI
				if self.alph[i2] > 0. and self.alph[i2] < self.C:
					if self.takeStep(i1, i2) == 1:
						return 1
			k0 = np.random.randint(0, self.endSupportI)
			for k in range(k0, k0 + self.endSupportI):
				i2 = k % self.endSupportI
				if self.takeStep(i1, i2) == 1:
					return 1
		return 0

	def takeStep(self, i1, i2):
		a1 = 0.
		a2 = 0.
		E1 = 0.
		E2 = 0.
		L = 0.
		H = 0.
		Lobj = 0.
		Hobj = 0.

		if i1 == i2:
			return 0
		alph1 = self.alph[i1]
		y1 = self.target[i1]
		if alph1 > 0. and alph1 < self.C:
			E1 = self.errorCache[i1]
		else:
			E1 = self.learnedFunc(i1) - y1
		alph2 = self.alph[i2]
		y2 = self.target[i2]
		if alph2 > 0. and alph2 < self.C:
			E2 = self.errorCache[i2]
		else:
			E2 = self.learnedFunc(i2) - y2
		s = y1 * y2
		if y1 == y2:
			gamma = alph1 + alph2
			if gamma > self.C:
				L = gamma - self.C
				H = self.C
			else:
				L = 0.
				H = gamma
		else:
			gamma = alph1 - alph2
			if gamma > 0.:
				L = 0.
				H = self.C - gamma
			else:
				L = -gamma
				H = self.C
		if L == H:
			return 0
		k11 = self.kernelFunc(i1, i1)
		k12 = self.kernelFunc(i1, i2)
		k22 = self.kernelFunc(i2, i2)
		eta = 2. * k12 - k11 - k22
		if eta < 0.:
			a2 = alph2 + y2 * (E2 - E1) / eta
			if a2 < L:
				a2 = L
			elif a2 > H:
				a2 = H
		else:
			c1 = eta / 2.
			c2 = y2 * (E1 - E2) - eta * alph2
			Lobj = c1 * L * L + c2 * L
			Hobj = c1 * H * H + c2 * H
			if Lobj > (Hobj + self.eps):
				a2 = L
			elif Lobj < (Hobj - self.eps):
				a2 = H
			else:
				a2 = alph2
		if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
			return 0
		a1 = alph1 - s * (a2 - alph2)
		if a1 < 0.:
			a2 = a2 + s * a1
			a1 = 0.
		elif a1 > self.C:
			t = a1 - self.C
			a2 = a2 + s * t
			a1 = self.C
		bnew = 0.
		if a1 > 0. and a1 < self.C:
			bnew = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
		elif a2 > 0. and a2 < self.C:
			bnew = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
		else:
			b1 = self.b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12
			b2 = self.b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22
			bnew = (b1 + b2) / 2.
		deltaB = bnew - self.b
		self.b = bnew
		t1 = y1 * (a1 - alph1)
		t2 = y2 * (a2 - alph2)
		for i in range(0, self.d):
			temp = self.densePoints[i1 * self.d + i] * t1 + self.densePoints[i2 * self.d + i] * t2
			temp1 = self.w[i]
			self.w[i] = temp + temp1
		t1 = y1 * (a1 - alph1)
		t2 = y2 * (a2 - alph2)
		for i in range(0, self.endSupportI):
			if self.alph[i] > 0. and self.alph[i] < self.C:
				tmp = self.errorCache[i]
				tmp = tmp + t1 * self.kernelFunc(i1, i) + t2 * self.kernelFunc(i2, i) - deltaB
				self.errorCache[i] = tmp
		self.errorCache[i1] = 0.
		self.errorCache[i2] = 0.
		self.alph[i1] = a1
		self.alph[i2] = a2
		return 1

	def errorRate(self):
		nTotal = 0
		nError = 0
		for i in range(self.firstTestI, self.N):
			if (self.learnedFunc(i) > 0.) != (self.target[i] > 0.):
				nError = nError + 1
			nTotal = nTotal + 1
		return 1. * nError / nTotal

	def testFunc(self, x):
		s = 0.
		for i in range(0, self.d):
			s = s + self.w[i] * x[i]
		s = s - self.b
		return s

	def learnedFunc(self, k):
		s = 0.
		for i in range(0, self.d):
			s = s + self.w[i] * self.densePoints[k * self.d + i]
		s = s - self.b
		return s

	def kernelFunc(self, i1, i2):
		dot = 0.
		for i in range(0, self.d):
			dot = dot + self.densePoints[i1 * self.d + i] * self.densePoints[i2 * self.d + i]
		return dot

def init(dataSize,jitMag):
	np.random.seed(10)
	svm = SVM()
	svm.nonlinear = 0
	svm.d = 2
	svm.N = dataSize
	svm.target = np.zeros(svm.N)
	svm.densePoints = np.zeros(svm.N * svm.d)
	svm.firstTestI = 0
	svm.endSupportI = svm.N
	svm.alph = np.zeros(svm.endSupportI)
	svm.b = 0.
	svm.errorCache = np.zeros(svm.N)
	svm.w = np.zeros(svm.d)
	trainCenter = np.random.rand(4)
	for dataI in range(0, svm.N):
		centerI = np.random.randint(2)
		if centerI == 0:
			svm.target[dataI] = -1
		else:
			svm.target[dataI] = 1
		for dimI in range(0, svm.d):
			svm.densePoints[dataI * svm.d + dimI] = np.random.randn() * jitMag + trainCenter[centerI * svm.d + dimI]
	return svm
def Training(svm):
	numChanged = 0
	examineAll = 1
	while numChanged > 0 or examineAll > 0:
		numChanged = 0
		if examineAll > 0:
			for k in range(0, svm.N):
				numChanged = numChanged + svm.examineExample(k)
		else:
			for k in range(0, svm.N):
				if svm.alph[k] != 0. and svm.alph[k] != svm.C:
					numChanged = numChanged + svm.examineExample(k)
		if examineAll == 1:
			examineAll = 0
		elif numChanged == 0:
			examineAll = 1
		non_bound_support = 0
		bound_support = 0
		for i in range(0, svm.N):
			if svm.alph[i] > 0.:
				if svm.alph[i] < svm.C:
					non_bound_support = non_bound_support + 1
				else:
					bound_support = bound_support + 1
	print('Threshold = ' + str(svm.b))
	print('Error rate = ' + str(svm.errorRate()))
	print(svm.w)
	return svm
def draw(svm,dataSize):
	fig = plt.figure(figsize = (10, 10))
	pointX = [[], []]
	pointY = [[], []]
	for i in range(0, dataSize):
		stateI = 0
		if svm.target[i] > 0.:
			stateI = 1
		else:
			stateI = 0
		pointX[stateI].append(svm.densePoints[i * 2])
		pointY[stateI].append(svm.densePoints[i * 2 + 1])
	plt.plot(pointX[0], pointY[0], 'o',  color = 'red')
	plt.plot(pointX[1], pointY[1], 'o',  color = 'green')
	plt.plot([-1., 2.], [(svm.b + svm.w[0]) / svm.w[1], (svm.b - 2. * svm.w[0]) / svm.w[1]],  color = 'blue')
	plt.title('SVM')
	plt.ylim(-1., 2.)
	plt.xlim(-1., 2.)
	plt.show()


def MyProgram(dataSize = 1500,jitMag = 0.1):
	svm = init(dataSize, jitMag)
	svm = Training(svm)
	draw(svm,dataSize)
	



MyProgram(1500, 0.1)