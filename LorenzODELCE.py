"""
Lorenz system Lyapunov exponent approximation and trajectory generation.

Based on:
- https://csc.ucdavis.edu/~chaos/courses/nlp/Software/partH.html
- https://csc.ucdavis.edu/~chaos/courses/nlp/Software/PartH_Code/LorenzODELCE.py
"""

import numpy as np

# The Lorenz 3D ODEs
#	Original parameter values: (sigma,R,b) = (10,28,-8/3)
def LorenzXDot(sigma,R,b,x,y,z):
	return sigma * (-x + y)

def LorenzYDot(sigma,R,b,x,y,z):
	return R*x - x*z - y

def LorenzZDot(sigma,R,b,x,y,z):
	return -b*z + x*y

# The tangent space (linearized) flow (aka co-tangent flow)
def LorenzDXDot(sigma,R,b,x,y,z,dx,dy,dz):
	return sigma * (-dx + dy)

def LorenzDYDot(sigma,R,b,x,y,z,dx,dy,dz):
	return (R-z)*dx - dy - x*dz

def LorenzDZDot(sigma,R,b,x,y,z,dx,dy,dz):
	return y*dx + x*dy - b*dz

# 3D fourth-order Runge-Kutta integrator
def RKThreeD(a,b,c,x,y,z,f,g,h,dt):
	k1x = dt * f(a,b,c,x,y,z)
	k1y = dt * g(a,b,c,x,y,z)
	k1z = dt * h(a,b,c,x,y,z)
	k2x = dt * f(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
	k2y = dt * g(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
	k2z = dt * h(a,b,c,x + k1x / 2.0,y + k1y / 2.0,z + k1z / 2.0)
	k3x = dt * f(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
	k3y = dt * g(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
	k3z = dt * h(a,b,c,x + k2x / 2.0,y + k2y / 2.0,z + k2z / 2.0)
	k4x = dt * f(a,b,c,x + k3x,y + k3y,z + k3z)
	k4y = dt * g(a,b,c,x + k3x,y + k3y,z + k3z)
	k4z = dt * h(a,b,c,x + k3x,y + k3y,z + k3z)
	x += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
	y += ( k1y + 2.0 * k2y + 2.0 * k3y + k4y ) / 6.0
	z += ( k1z + 2.0 * k2z + 2.0 * k3z + k4z ) / 6.0
	return x,y,z

# Tanget space flow (using fourth-order Runge-Kutta integrator)
def TangentFlowRKThreeD(a,b,c,x,y,z,df,dg,dh,dx,dy,dz,dt):
	k1x = dt * df(a,b,c,x,y,z,dx,dy,dz)
	k1y = dt * dg(a,b,c,x,y,z,dx,dy,dz)
	k1z = dt * dh(a,b,c,x,y,z,dx,dy,dz)
	k2x = dt * df(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
	k2y = dt * dg(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
	k2z = dt * dh(a,b,c,x,y,z,dx+k1x/2.0,dy+k1y/2.0,dz+k1z/2.0)
	k3x = dt * df(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
	k3y = dt * dg(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
	k3z = dt * dh(a,b,c,x,y,z,dx+k2x/2.0,dy+k2y/2.0,dz+k2z/2.0)
	k4x = dt * df(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
	k4y = dt * dg(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
	k4z = dt * dh(a,b,c,x,y,z,dx+k3x,dy+k3y,dz+k3z)
	dx += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
	dy += ( k1y + 2.0 * k2y + 2.0 * k3y + k4y ) / 6.0
	dz += ( k1z + 2.0 * k2z + 2.0 * k3z + k4z ) / 6.0
	return dx,dy,dz

def lorenz_lce(
	dt=0.01,  # integration time step
	nTransients=100,  # iterates to ignore for trajectory
	nIterates=1000,  # time steps to integrate over for trajectory
	nTransients_lce=100,  # iterates to ignore for lce estimation
	nIterates_lce=10000,  # time steps to integrate over for lce estimation
	nItsPerPB=10,  # time steps for pullback calculation
	lorenzParams=dict(sigma=10.0, R=28.0, b=8/3),  # control parameters
	lorenzState=dict(x=5.0, y=5.0, z=5.0),  # initial state
	includeTrajectory=False,  # return the trajectory in the result
	fullLceSpectrum=False,  # return all Lyapunov exponents (default just the largest)
	):

	sigma, R, b = lorenzParams["sigma"], lorenzParams["R"], lorenzParams["b"]
	xState, yState, zState = lorenzState["x"], lorenzState["y"], lorenzState["z"]

	if includeTrajectory:

		# Iterate for some number of transients, but don't use these states
		for n in range(0,nTransients):
			xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,LorenzXDot,LorenzYDot,LorenzZDot,dt)

		# Set up array of iterates and store the current state
		x = [xState]
		y = [yState]
		z = [zState]

		for n in range(0,nIterates):
			# at each time step calculate new (x,y,z)(t)
			xt,yt,zt = RKThreeD(sigma,R,b,x[n],y[n],z[n],LorenzXDot,LorenzYDot,LorenzZDot,dt)
			# and append to lists
			x.append(xt)
			y.append(yt)
			z.append(zt)

	# Compute the LCE spectrum
	# Initial tangent vectors
	e1x = 1.0
	e1y = 0.0
	e1z = 0.0
	e2x = 0.0
	e2y = 1.0
	e2z = 0.0
	e3x = 0.0
	e3y = 0.0
	e3z = 1.0

	xState, yState, zState = lorenzState["x"], lorenzState["y"], lorenzState["z"]  # reinitialise the state

	# Iterate away transients and let the tangent vectors align
	#	with the global stable and unstable manifolds
	for n in range(0,nTransients_lce):
		for i in range(nItsPerPB):
			xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,\
				LorenzXDot,LorenzYDot,LorenzZDot,dt)
			# Evolve tangent vector for maximum LCE (LCE1)
			e1x,e1y,e1z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
				LorenzDXDot,LorenzDYDot,LorenzDZDot,e1x,e1y,e1z,dt)
			
			if fullLceSpectrum:
				# Evolve tangent vector for next LCE (LCE2)
				e2x,e2y,e2z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
					LorenzDXDot,LorenzDYDot,LorenzDZDot,e2x,e2y,e2z,dt)
				# Evolve tangent vector for last LCE
				e3x,e3y,e3z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
					LorenzDXDot,LorenzDYDot,LorenzDZDot,e3x,e3y,e3z,dt)

		# Normalize the tangent vector
		d = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
		e1x /= d
		e1y /= d
		e1z /= d

		if fullLceSpectrum:
			# Pull-back: Remove any e1 component from e2
			dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
			e2x -= dote1e2 * e1x
			e2y -= dote1e2 * e1y
			e2z -= dote1e2 * e1z
			# Normalize second tangent vector
			d = np.sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
			e2x /= d
			e2y /= d
			e2z /= d
			# Pull-back: Remove any e1 and e2 components from e3
			dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
			dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
			e3x -= dote1e3 * e1x + dote2e3 * e2x
			e3y -= dote1e3 * e1y + dote2e3 * e2y
			e3z -= dote1e3 * e1z + dote2e3 * e2z
			# Normalize third tangent vector
			d = np.sqrt(e3x*e3x + e3y*e3y + e3z*e3z)
			e3x /= d
			e3y /= d
			e3z /= d

	# Begin the estimation
	LCE1 = 0.0

	if fullLceSpectrum:
		LCE2 = 0.0
		LCE3 = 0.0

	for n in range(0,nIterates_lce):
		for i in range(nItsPerPB):
			xState,yState,zState = RKThreeD(sigma,R,b,xState,yState,zState,\
				LorenzXDot,LorenzYDot,LorenzZDot,dt)
			# Evolve tangent vector for maximum LCE (LCE1)
			e1x,e1y,e1z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
				LorenzDXDot,LorenzDYDot,LorenzDZDot,e1x,e1y,e1z,dt)

			if fullLceSpectrum:
				# Evolve tangent vector for next LCE (LCE2)
				e2x,e2y,e2z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
					LorenzDXDot,LorenzDYDot,LorenzDZDot,e2x,e2y,e2z,dt)
				# Evolve tangent vector for last LCE
				e3x,e3y,e3z = TangentFlowRKThreeD(sigma,R,b,xState,yState,zState,\
					LorenzDXDot,LorenzDYDot,LorenzDZDot,e3x,e3y,e3z,dt)

		# Normalize the tangent vector
		d = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
		e1x /= d
		e1y /= d
		e1z /= d
		# Accumulate the first tangent vector's length change factor
		LCE1 += np.log(d)

		if fullLceSpectrum:
			# Pull-back: Remove any e1 component from e2
			dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
			e2x -= dote1e2 * e1x
			e2y -= dote1e2 * e1y
			e2z -= dote1e2 * e1z
			# Normalize second tangent vector
			d = np.sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
			e2x /= d
			e2y /= d
			e2z /= d
			# Accumulate the second tangent vector's length change factor
			LCE2 += np.log(d)
			# Pull-back: Remove any e1 and e2 components from e3
			dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
			dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
			e3x -= dote1e3 * e1x + dote2e3 * e2x
			e3y -= dote1e3 * e1y + dote2e3 * e2y
			e3z -= dote1e3 * e1z + dote2e3 * e2z
			# Normalize third tangent vector
			d = np.sqrt(e3x*e3x + e3y*e3y + e3z*e3z)
			e3x /= d
			e3y /= d
			e3z /= d
			# Accumulate the third tangent vector's length change factor
			LCE3 += np.log(d)

	# Convert to per-iterate, per-second LCEs
	IntegrationTime = dt * float(nItsPerPB) * float(nIterates_lce)
	LCE1 = LCE1 / IntegrationTime

	if fullLceSpectrum:
		LCE2 = LCE2 / IntegrationTime
		LCE3 = LCE3 / IntegrationTime

	result = dict()
	result["system"] = "lorenz"
	result["params"] = lorenzParams
	result["initial"] = lorenzState
	result["iterates"] = dict(
		trajectory={"nTransients":nTransients, "nIterates":nIterates},
		lce={"nTransients":nTransients_lce, "nIterates":nIterates_lce, "nItsPerPB":nItsPerPB}
		)
	result["lce"] = (LCE1, LCE2, LCE3) if fullLceSpectrum else (LCE1,)
	result["trajectory"] = np.array([xyz for xyz in zip(x, y, z)]) if includeTrajectory else np.array([])

	return result

if __name__ == "__main__":
	rr = np.sort(np.random.uniform(0,100,2))
	lces = [lorenz_lce(lorenzParams=dict(sigma=10, R=R, b=8/3), includeTrajectory=True, fullLceSpectrum=True, nIterates=1000) for R in rr]
	print(lces[0])
