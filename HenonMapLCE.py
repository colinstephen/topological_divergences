"""
Hénon system Lyapunov exponent approximation and trajectory generation.

Based on:
- https://csc.ucdavis.edu/~chaos/courses/nlp/Software/partH.html
- https://csc.ucdavis.edu/~chaos/courses/nlp/Software/PartH_Code/HenonMapLCE.py
"""

import numpy as np

def HenonMap(a,b,x,y):
	return y + 1.0 - a *x*x, b * x

def HenonMapTangent(a,b,x,y,dx,dy):
	return dy - 2.0 * a * x * dx, b * dx

def henon_lce(
	henonParams = dict(a=1.4, b=0.3),
	henonState = dict(x=0.1, y=0.3),
	nTransients = 200,
	nIterates = 10000,
	includeTrajectory = False,
	fullLceSpectrum = False,
):
	a, b = henonParams["a"], henonParams["b"]
	xState, yState = henonState["x"], henonState["y"]

	if includeTrajectory:

		# Throw away transients, so we're on an attractor
		for n in range(0,nTransients):
			xState, yState = HenonMap(a,b,xState,yState)

		# Set up arrays of iterates (x_n,y_n) and set the initial condition
		x = [xState]
		y = [yState]

		# The main loop that generates iterates and stores them for plotting
		for n in range(0,nIterates):
			# at each iteration calculate (x_n+1,y_n+1)
			xState, yState = HenonMap(a,b,x[n],y[n])
			# and append to lists x and y
			x.append( xState )
			y.append( yState )

	# Initial condition
	xState, yState = henonState["x"], henonState["y"]

	# Initial tangent vectors
	e1x = 1.0
	e1y = 0.0
	e2x = 0.0
	e2y = 1.0

	# Iterate away transients and let the tangent vectors align
	#    with the global stable and unstable manifolds
	for n in range(0,nTransients):
		xState, yState = HenonMap(a,b,xState,yState)

		# Evolve tangent vector for maxLCE
		e1x, e1y = HenonMapTangent(a,b,xState,yState,e1x,e1y)
	
		# Normalize the tangent vector's length
		d = np.sqrt(e1x*e1x + e1y*e1y)
		e1x = e1x / d
		e1y = e1y / d
	
		if fullLceSpectrum:
			# Evolve tangent vector for minLCE
			e2x, e2y = HenonMapTangent(a,b,xState,yState,e2x,e2y)
		
			# Pull-back: Remove any e1 component from e2
			dote1e2 = e1x * e2x + e1y * e2y
			e2x = e2x - dote1e2 * e1x
			e2y = e2y - dote1e2 * e1y
		
			# Normalize second tangent vector
			d = np.sqrt(e2x*e2x + e2y*e2y)
			e2x = e2x / d
			e2y = e2y / d

	# Okay, now we're ready to begin the estimation
	# This is essentially the same as above, except we accumulate estimates
	# We have to set the min,max LCE estimates to zero, since they are sums
	maxLCE = 0.0
	if fullLceSpectrum:
		minLCE = 0.0

	for n in range(0,nIterates):
		# Get next state
		xState, yState = HenonMap(a,b,xState,yState)
		
		# Evolve tangent vector for maxLCE
		e1x, e1y = HenonMapTangent(a,b,xState,yState,e1x,e1y)
	
		# Normalize the tangent vector's length
		d = np.sqrt(e1x*e1x + e1y*e1y)
		e1x = e1x / d
		e1y = e1y / d
		
		# Accumulate the stretching factor (tangent vector's length)
		maxLCE = maxLCE + np.log(d)

		if fullLceSpectrum:
			# Evolve tangent vector for minLCE
			e2x, e2y = HenonMapTangent(a,b,xState,yState,e2x,e2y)
			
			# Pull-back: Remove any e1 component from e2
			dote1e2 = e1x * e2x + e1y * e2y
			e2x = e2x - dote1e2 * e1x
			e2y = e2y - dote1e2 * e1y
			
			# Normalize second tangent vector
			d = np.sqrt(e2x*e2x + e2y*e2y)
			e2x = e2x / d
			e2y = e2y / d
			
			# Accumulate the shrinking factor (tangent vector's length)
			minLCE = minLCE + np.log(d)

	# Convert to per-iterate LCEs
	maxLCE = maxLCE / float(nIterates)
	if fullLceSpectrum:
		minLCE = minLCE / float(nIterates)

	result = dict()
	result["params"] = henonParams
	result["initial"] = henonState
	result["lce"] = (maxLCE, minLCE) if fullLceSpectrum else (maxLCE,)
	result["trajectory"] = np.array([xy for xy in zip(x, y)]) if includeTrajectory else np.array([])

	return result

if __name__ == "__main__":
	aa = np.sort(np.random.uniform(0.8, 1.4, 10))
	lces = [henon_lce(henonParams=dict(a=a, b=0.3), includeTrajectory=True, fullLceSpectrum=True) for a in aa]
	print(lces[0])
