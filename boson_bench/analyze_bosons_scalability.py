import time
import sys
import logging
import re
import statistics
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools
import tempfile
from scipy import stats
import math
import os
import argparse

def analyze_scalability_csv(infile_path):
    data = np.genfromtxt(infile_path, delimiter=",", skip_header=True)
    data_log = np.log2(data)

    keyfunc = lambda d: d[0]
    nbosons = []
    times_per_nbosons = []
    data = sorted(data, key=keyfunc)
    for k, g in itertools.groupby(data, keyfunc):
        nbosons.append(k)
        times_per_nbosons.append(np.asarray(list(g))[:,1])

    x = np.log2(np.asarray(nbosons))
    logtimes_per_boson = np.log2(times_per_nbosons)
    y = np.mean(logtimes_per_boson, axis=1)
    err = np.std(logtimes_per_boson, axis=1)
    print("std:", err)

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    print(slope, intercept, r, p)

    plt.errorbar(x, y, err, linestyle='None', marker='o', ecolor='red')
    plt.plot(x, intercept + slope * x, '-', label=f'slope ${slope:.3f}$')
    plt.xlabel('N')
    plt.xticks(x, [int(n) for n in nbosons])
    plt.ylabel('time (s)')
    plt.gca().yaxis.set_major_formatter(lambda y, pos: str(int(pow(2, y))))
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='analyze boson scalability results')

    parser.add_argument('infile', type=str,
                        help='path to csv with results')
    args = parser.parse_args()

    analyze_scalability_csv(args.infile)

if __name__ == "__main__":
    main()
