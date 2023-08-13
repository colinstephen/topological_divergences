import numpy as np

def LogisticMap(r,x):
    return r * x * (1 - x)

def LogisticMapTangent(r, x, dx):
    return r - 2 * r * x

def logistic_lce(
    logisticParams = dict(r=4.0),
    logisticState = dict(x=0.2),
    nTransients = 200,
    nIterates = 10000,
    includeTrajectory = False,
):
    r = logisticParams["r"]
    xState = logisticState["x"]

    if includeTrajectory:

        for n in range(0, nTransients):
            xState = LogisticMap(r,xState)

        x = [xState]

        for n in range(0, nIterates):
            xState = LogisticMap(r,x[n])
            x.append( xState )

    xState = logisticState["x"]

    # Initial tangent vector
    e1x = 1.0

    for n in range(0, nTransients):
        xState = LogisticMap(r, xState)

        # Evolve tangent vector for LCE
        e1x = LogisticMapTangent(r, xState, e1x)

        # Normalize the tangent vector's length
        d = np.sqrt(e1x*e1x)
        e1x = e1x / d

    LCE = 0.0

    for n in range(0, nIterates):
        xState = LogisticMap(r, xState)

        # Evolve tangent vector for LCE
        e1x = LogisticMapTangent(r, xState, e1x)

        # Normalize the tangent vector's length
        d = np.sqrt(e1x*e1x)
        e1x = e1x / d

        # Accumulate the stretching factor (tangent vector's length)
        LCE = LCE + np.log(d)

    # Convert to per-iterate LCE
    LCE = LCE / float(nIterates)

    result = dict()
    result["params"] = logisticParams
    result["initial"] = logisticState
    result["lce"] = (LCE,)
    result["trajectory"] = np.array(x) if includeTrajectory else np.array([])

    return result

if __name__ == "__main__":
    rr = np.sort(np.random.uniform(3,4,10))
    lces = [logistic_lce(logisticParams=dict(r=r), includeTrajectory=True) for r in rr]
    print(lces[0])