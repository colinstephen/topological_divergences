import numpy as np
import matplotlib.pyplot as plt


def merge_tree(time_series_data):
    n = len(time_series_data)
    data = sorted([(x, i) for i, x in enumerate(time_series_data)], reverse=True)

    # each node starts as its own parent (root)
    parent = list(range(n))

    def find_root(i):
        if i == parent[i]:
            return i
        parent[i] = find_root(parent[i])  # path compression
        return parent[i]

    def union(i, j):
        ri, rj = find_root(i), find_root(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)  # merge into the smaller root

    # process points from high to low
    for x, i in data:
        if i > 0 and time_series_data[i - 1] >= x:
            union(i, i - 1)
        if i + 1 < n and time_series_data[i + 1] >= x:
            union(i, i + 1)

    # now find the roots for each node
    roots = [find_root(i) for i in range(n)]

    return roots


time_series_data = np.array([1, 5, 2, 8, 3, 6, 9, 7, 4, 0])
roots = merge_tree(time_series_data)

plt.plot(time_series_data)
for i, r in enumerate(roots):
    if i != r:
        plt.plot([i, r], [time_series_data[i], time_series_data[r]], "r-")
plt.show()
