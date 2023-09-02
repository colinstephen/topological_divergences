# dynamic system trajectories and data for use in lyapunov estimation

import numpy as np

from functools import partial

from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence

from LogisticMapLCE import logistic_lce
from HenonMapLCE import henon_lce
from IkedaMapLCE import ikeda_lce
from TinkerbellMapLCE import tinkerbell_lce

import ipyparallel as ipp
clients = ipp.Client()
dv = clients.direct_view()
lbv = clients.load_balanced_view()

def configdict(cls):
    """Given a configuration class, convert it and its properties to a dictionary."""
    attributes = {
        attr: getattr(cls, attr)
        for attr in dir(cls)
        if not callable(getattr(cls, attr)) and not attr.startswith("__")
    }
    return {cls.__name__: attributes}


def z_normalise(ts: np.array) -> np.array:
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std


def generate_trajectories(RANDOM_SEED=42, TS_LENGTH=500, CONTROL_PARAM_SAMPLES=500):

    print(f"Experiment config -- SEED:{RANDOM_SEED}, LENGTH:{TS_LENGTH}, SAMPLES:{CONTROL_PARAM_SAMPLES}")

    class EXPERIMENT_CONFIG:
        SEED = RANDOM_SEED
        RANDOM_STATE = RandomState(MT19937(SeedSequence(SEED)))
        TIME_SERIES_LENGTH = TS_LENGTH
        NUM_CONTROL_PARAM_SAMPLES = CONTROL_PARAM_SAMPLES

    class LOGISTIC_CONFIG:
        R_MIN = 3.5
        R_MAX = 4.0
        TRAJECTORY_DIM = 0

    logistic_control_params = [
        dict(r=r)
        for r in np.sort(
            EXPERIMENT_CONFIG.RANDOM_STATE.uniform(
                LOGISTIC_CONFIG.R_MIN,
                LOGISTIC_CONFIG.R_MAX,
                EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES,
            )
        )
    ]
    logistic_lce_partial = partial(logistic_lce, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    logistic_dataset = lbv.map_sync(logistic_lce_partial, logistic_control_params)
    # logistic_dataset = [
    #     logistic_lce(
    #         mapParams=params,
    #         nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH,
    #         includeTrajectory=True,
    #     )
    #     for params in logistic_control_params
    # ]
    logistic_trajectories = [
        z_normalise(data["trajectory"][:, LOGISTIC_CONFIG.TRAJECTORY_DIM])
        for data in logistic_dataset
    ]
    logistic_lces = np.array([data["lce"][0] for data in logistic_dataset])

    logistic_control_param_values = [params["r"] for params in logistic_control_params]

    class HENON_CONFIG:
        A_MIN = 0.8
        A_MAX = 1.4
        B = 0.3
        TRAJECTORY_DIM = 0

    henon_control_params = [
        dict(a=a, b=HENON_CONFIG.B)
        for a in np.sort(
            EXPERIMENT_CONFIG.RANDOM_STATE.uniform(
                HENON_CONFIG.A_MIN,
                HENON_CONFIG.A_MAX,
                EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES,
            )
        )
    ]
    henon_lce_partial = partial(henon_lce, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    henon_dataset = lbv.map_sync(henon_lce_partial, henon_control_params)
    # henon_dataset = [
    #     henon_lce(
    #         mapParams=params,
    #         nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH,
    #         includeTrajectory=True,
    #     )
    #     for params in henon_control_params
    # ]
    henon_trajectories = [
        z_normalise(data["trajectory"][:, HENON_CONFIG.TRAJECTORY_DIM])
        for data in henon_dataset
    ]
    henon_lces = np.array([data["lce"][0] for data in henon_dataset])

    henon_control_param_values = [params["a"] for params in henon_control_params]

    class IKEDA_CONFIG:
        A_MIN = 0.5
        A_MAX = 1.0
        TRAJECTORY_DIM = 0

    ikeda_control_params = [
        dict(a=a)
        for a in np.sort(
            EXPERIMENT_CONFIG.RANDOM_STATE.uniform(
                IKEDA_CONFIG.A_MIN,
                IKEDA_CONFIG.A_MAX,
                EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES,
            )
        )
    ]
    ikeda_lce_partial = partial(ikeda_lce, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    ikeda_dataset = lbv.map_sync(ikeda_lce_partial, ikeda_control_params)
    # ikeda_dataset = [
    #     ikeda_lce(
    #         mapParams=params,
    #         nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH,
    #         includeTrajectory=True,
    #     )
    #     for params in ikeda_control_params
    # ]
    ikeda_trajectories = [
        z_normalise(data["trajectory"][:, IKEDA_CONFIG.TRAJECTORY_DIM])
        for data in ikeda_dataset
    ]
    ikeda_lces = np.array([data["lce"][0] for data in ikeda_dataset])

    ikeda_control_param_values = [params["a"] for params in ikeda_control_params]

    class TINKERBELL_CONFIG:
        A_MIN = 0.7
        A_MAX = 0.9
        TRAJECTORY_DIM = 0

    tinkerbell_control_params = [
        dict(a=a)
        for a in np.sort(
            EXPERIMENT_CONFIG.RANDOM_STATE.uniform(
                TINKERBELL_CONFIG.A_MIN,
                TINKERBELL_CONFIG.A_MAX,
                EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES,
            )
        )
    ]
    tinkerbell_lce_partial = partial(tinkerbell_lce, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    tinkerbell_dataset = lbv.map_sync(tinkerbell_lce_partial, tinkerbell_control_params)
    # tinkerbell_dataset = [
    #     tinkerbell_lce(
    #         mapParams=params,
    #         nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH,
    #         includeTrajectory=True,
    #     )
    #     for params in tinkerbell_control_params
    # ]
    tinkerbell_trajectories = [
        z_normalise(data["trajectory"][:, TINKERBELL_CONFIG.TRAJECTORY_DIM])
        for data in tinkerbell_dataset
    ]
    tinkerbell_lces = np.array([data["lce"][0] for data in tinkerbell_dataset])

    tinkerbell_control_param_values = [
        params["a"] for params in tinkerbell_control_params
    ]

    datasets = {
        "logistic": {
            "exp_config": configdict(EXPERIMENT_CONFIG),
            "sys_config": configdict(LOGISTIC_CONFIG),
            "sys_params": logistic_control_param_values,
            "param_name": "r",
            "dataset": logistic_dataset,
            "trajectories": logistic_trajectories,
            "lces": logistic_lces,
        },
        "ikeda": {
            "exp_config": configdict(EXPERIMENT_CONFIG),
            "sys_config": configdict(IKEDA_CONFIG),
            "sys_params": ikeda_control_param_values,
            "param_name": "a",
            "dataset": ikeda_dataset,
            "trajectories": ikeda_trajectories,
            "lces": ikeda_lces,
        },
        "henon": {
            "exp_config": configdict(EXPERIMENT_CONFIG),
            "sys_config": configdict(HENON_CONFIG),
            "sys_params": henon_control_param_values,
            "param_name": "a",
            "dataset": henon_dataset,
            "trajectories": henon_trajectories,
            "lces": henon_lces,
        },
        "tinkerbell": {
            "exp_config": configdict(EXPERIMENT_CONFIG),
            "sys_config": configdict(TINKERBELL_CONFIG),
            "sys_params": tinkerbell_control_param_values,
            "param_name": "a",
            "dataset": tinkerbell_dataset,
            "trajectories": tinkerbell_trajectories,
            "lces": tinkerbell_lces,
        },
    }

    return datasets
