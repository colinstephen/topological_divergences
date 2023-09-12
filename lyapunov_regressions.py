"""Using topological divergences to estimate maximum Lyapunov exponents."""

import pickle
import logging
from datetime import datetime as dt
from functools import partial
from TimeSeriesMergeTreeSimple import TimeSeriesMergeTree as TSMT
from ipyparallel import require
from sklearn.preprocessing import MinMaxScaler
from tree_offset_divergence import get_offset_divergences
from trajectories import generate_trajectories as ts_data
import ipyparallel as ipp

logging.basicConfig(level=logging.INFO)


clients = ipp.Client()
dv = clients.direct_view()
lbv = clients.load_balanced_view()


## Set parameters for the experiment

SEED = 42
SAMPLES = 500
LENGTH = 500


## Set up the data

logging.info(f"Starting to generate system data at {dt.utcnow()}")
system_data = ts_data(
    RANDOM_SEED=SEED,
    TS_LENGTH=LENGTH,
    CONTROL_PARAM_SAMPLES=SAMPLES,
    normalise=False,
)
logging.info(f"Finished generating system data at {dt.utcnow()}")

def scale(ts):
    """Make range of ts fall between 0 and 1"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(ts.reshape(-1, 1)).flatten()

for system in system_data:
    trajectories = system_data[system]["trajectories"]
    trajectories = list(map(scale, trajectories))
    system_data[system]["trajectories"] = trajectories


## Create the representations and extract their divergences

for system in system_data:
    logging.info(f"Beginning to process {system} data at {dt.utcnow()}")
    ddict = system_data[system]

    ddict["plmt"] = {}
    ddict["dmt"] = {}

    ddict["plmt"]["func"] = partial(TSMT, discrete=False)
    ddict["dmt"]["func"] = partial(TSMT, discrete=True)

    ddict["plmt"]["reps"] = list(map(ddict["plmt"]["func"], system_data[system]["trajectories"]))
    ddict["dmt"]["reps"] = list(map(ddict["dmt"]["func"], system_data[system]["trajectories"]))

    ddict["plmt"]["offsets"] = range(1,101,25)
    ddict["dmt"]["offsets"] = range(1,101,25)

    @require(partial=partial, get_offset_divergences=get_offset_divergences, offsets=ddict["plmt"]["offsets"])
    def divs_for_merge_tree(merge_tree):
        return list(map(partial(get_offset_divergences, tsmt=merge_tree), offsets))
    ddict["plmt"]["divs"] = lbv.map_sync(divs_for_merge_tree, ddict["plmt"]["reps"])
    logging.info(f"Finished computing divergences for PLMT at {dt.utcnow()}")

    @require(partial=partial, get_offset_divergences=get_offset_divergences, offsets=ddict["dmt"]["offsets"])
    def divs_for_merge_tree(merge_tree):
        return list(map(partial(get_offset_divergences, tsmt=merge_tree), offsets))
    ddict["dmt"]["divs"] = lbv.map_sync(divs_for_merge_tree, ddict["dmt"]["reps"])
    logging.info(f"Finished computing divergences for DMT at {dt.utcnow()}")

    with open(f"outputs/data/TESTING_lyapunov_regression_for_{system}.pkl", "wb") as file:
        pickle.dump(ddict, file)
        logging.info(f"Saved system info for {system} at {dt.utcnow()}.")
