"""
Tinkerbell map Lyapunov exponent approximation and trajectory generation.
"""

import numpy as np

def TinkerbellMap(a,x,y):
	return x*x - y*y + a*x - 0.6013*y, 2*x*y + 2*x + 0.5*y

def TinkerbellMapTangent(a,x,y,dx,dy):
	Dx = (2*x + a)*dx - (2*y + 0.6013)*dy
	Dy = (2*y + 2)*dx + (2*x + 0.5)*dy
	return Dx, Dy

def tinkerbell_lce(
	mapParams = dict(a=0.9),
	initialState = dict(x=0.1, y=0.1),
	nTransients = 100,
	nIterates = 1000,
	nTransients_lce = 200,
	nIterates_lce = 10000,
	includeTrajectory = False,
	fullLceSpectrum = False,
):
	a = mapParams["a"]
	xState, yState = initialState["x"], initialState["y"]

	if includeTrajectory:

		# Throw away transients, so we're on an attractor
		for n in range(0,nTransients):
			xState, yState = TinkerbellMap(a,xState,yState)

		# Set up arrays of iterates (x_n,y_n) and set the initial condition
		x = [xState]
		y = [yState]

		# The main loop that generates iterates and stores them for plotting
		for n in range(0,nIterates):
			# at each iteration calculate (x_n+1,y_n+1)
			xState, yState = TinkerbellMap(a,x[n],y[n])
			# and append to lists x and y
			x.append( xState )
			y.append( yState )

	# Initial condition
	xState, yState = initialState["x"], initialState["y"]

	# Initial tangent vectors
	e1x = 1.0
	e1y = 0.0
	e2x = 0.0
	e2y = 1.0

	# Iterate away transients and let the tangent vectors align
	#    with the global stable and unstable manifolds
	for n in range(0,nTransients_lce):
		xState, yState = TinkerbellMap(a,xState,yState)

		# Evolve tangent vector for maxLCE
		e1x, e1y = TinkerbellMapTangent(a,xState,yState,e1x,e1y)
	
		# Normalize the tangent vector's length
		d = np.sqrt(e1x*e1x + e1y*e1y)
		e1x = e1x / d
		e1y = e1y / d
	
		if fullLceSpectrum:
			# Evolve tangent vector for minLCE
			e2x, e2y = TinkerbellMapTangent(a,xState,yState,e2x,e2y)
		
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

	for n in range(0,nIterates_lce):
		# Get next state
		xState, yState = TinkerbellMap(a,xState,yState)
		
		# Evolve tangent vector for maxLCE
		e1x, e1y = TinkerbellMapTangent(a,xState,yState,e1x,e1y)
	
		# Normalize the tangent vector's length
		d = np.sqrt(e1x*e1x + e1y*e1y)
		e1x = e1x / d
		e1y = e1y / d
		
		# Accumulate the stretching factor (tangent vector's length)
		maxLCE = maxLCE + np.log(d)

		if fullLceSpectrum:
			# Evolve tangent vector for minLCE
			e2x, e2y = TinkerbellMapTangent(a,xState,yState,e2x,e2y)
			
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
	maxLCE = maxLCE / float(nIterates_lce)
	if fullLceSpectrum:
		minLCE = minLCE / float(nIterates_lce)

	result = dict()
	result["system"] = "tinkerbell"
	result["params"] = mapParams
	result["initial"] = initialState
	result["iterates"] = dict(
		trajectory={"nTransients":nTransients, "nIterates":nIterates},
		lce={"nTransients":nTransients_lce, "nIterates":nIterates_lce}
		)
	result["lce"] = (maxLCE, minLCE) if fullLceSpectrum else (maxLCE,)
	result["trajectory"] = np.array([xy for xy in zip(x, y)]) if includeTrajectory else np.array([])

	return result

if __name__ == "__main__":
	aa = np.sort(np.random.uniform(0.7, 0.9, 10))
	lces = [tinkerbell_lce(mapParams=dict(a=a), includeTrajectory=True, fullLceSpectrum=True) for a in aa]
	print(lces[0])
	print([lce["lce"][0] for lce in lces])
