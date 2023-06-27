from tsph import compute_persistent_homology


def join_ordered_barcodes(barcode1, barcode2):
    merged_barcode = list(barcode1) + list(barcode2)
    joined_barcode = []

    for pair in merged_barcode:
        if not joined_barcode or pair[0] > joined_barcode[-1][1]:
            joined_barcode.append(pair)
        elif pair[1] > joined_barcode[-1][1]:
            joined_barcode[-1] = (joined_barcode[-1][0], pair[1])

    return joined_barcode


time_series = [4, 2, 6, 4, 3, 5, 7, 2, 3, 1, 6, 5, 7, 6, 3, 8]

pd = compute_persistent_homology(time_series)
pd1 = compute_persistent_homology(time_series[:8])
pd2 = compute_persistent_homology(time_series[8:])

print(pd)
print(pd1)
print(pd2)

joined_pd = join_ordered_barcodes(pd1, pd2)
print(joined_pd)
