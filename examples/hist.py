from __future__ import print_function
from __future__ import division

def histogram(data, n_bins=10):
    ''' Create an histogram
        :param data: A list with all values
        :n_bins: Classes of data
    '''

    # Find the minimum value
    min_num = min(data)

    # Find the maximum value
    max_num = max(data)

    # Compute the range of each bin
    gap = (max_num - min_num)/n_bins

    # Compute the limits of bins 
    bin_lims = []
    for bin_el in range(n_bins):
        bin_lims.append([
            min_num + bin_el * gap,
            min_num + (bin_el + 1) * gap])

    # Compute the frequence for each bin
    freqs = {}
    for el in data:
        for i, lims in enumerate(bin_lims):
            if lims[0] <= el < lims[1]:
                if i in freqs.keys():
                    freqs[i] += 1
                else:
                    freqs[i] = 1

    # Sum of frequencies
    tot = sum(freqs.values())

    # Plot the histogram
    for idx, freq in freqs.items():
        # Compute the proportion in each bin
        prop = freq / tot  
        # Each star in the string is 1% of values 
        stars = ("*" * int(100*(prop))) 
        # Put together all params for printing
        els = bin_lims[idx] + [freq, stars]  
        # It must be a tuple (not a list)
        els = tuple(els)
        # Fill the format string and print it
        print("%5.2f <-> %5.2f: %#3d %s" % els)

    return freqs, bin_lims

if __name__ == "__main__":

    # Load data from file
    data = []
    with open("hist_data.txt", "r") as datafile:
        for line in  datafile.readlines():
            data.append(float(line))

    # make histogram
    histogram(data)

