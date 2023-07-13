import time
import numpy as np
from multiprocessing import Pool, cpu_count

print("CPU count:", cpu_count())

def compute_value(args):
    i, j = args

    # Perform CPU-intensive computation
    start_time = time.time()
    while time.time() - start_time < 0.1:
        pass
    
    return i + j

def populate_array_parallel(arr, num_processes=None):

    rows, cols = arr.shape

    if num_processes is None:
        num_processes = cpu_count()  # Use all available CPUs

    with Pool(processes=num_processes) as pool:
        indices = [(i, j) for i in range(rows) for j in range(cols)]
        results = pool.map(compute_value, indices)

    for (i, j), value in zip(indices, results):
        arr[i, j] = value

# Example usage
rows, cols = 500, 500
arr = np.empty((rows, cols), dtype=np.float64)
num_processes = 127  # Set the number of processes here

run = 0
while True:
    run += 1
    start = time.time()
    print(f"Run {run} started at time {start}")
    populate_array_parallel(arr, num_processes)
    end = time.time()
    print(f"Run {run} ended at time {end} with duration {start - end}")
