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

def analyze_scalability_csv():
    data = np.genfromtxt(BOSON_SCALING_CSV_OUTPUT_PATH, delimiter=",", skip_header=True)
    data_log = np.log2(data)
    x = data_log[:,0]
    y = data_log[:,1]
    # err = data[:,2] / data[:,0]
    err = np.zeros(len(x))

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    print(slope, intercept, r, p)

    plt.errorbar(x, y, err, linestyle='None', marker='o', ecolor='red')
    plt.plot(x, intercept + slope * x, '-')
    plt.xlabel('log N')
    plt.ylabel('log time (s)')
    plt.show()

def analyze_scalability_csv_raw(infile_path):
    data = np.genfromtxt(infile_path, delimiter=",", skip_header=True)
    data_log = np.log2(data)

    keyfunc = lambda d: d[0]
    nbosons = []
    times_per_nbosons = []
    data = sorted(data_log, key=keyfunc)
    for k, g in itertools.groupby(data, keyfunc):
        nbosons.append(k)
        times_per_nbosons.append(np.asarray(list(g))[:,1])

    x = np.asarray(nbosons)
    y = np.mean(times_per_nbosons, axis=1)
    err = np.std(times_per_nbosons, axis=1)
    print("std:", err)

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    print(slope, intercept, r, p)

    plt.errorbar(x, y, err, linestyle='None', marker='o', ecolor='red')
    plt.plot(x, intercept + slope * x, '-')
    plt.xlabel('log N')
    plt.ylabel('log time (s)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='analyze boson scalability results')

    parser.add_argument('infile', type=str,
                        help='path to csv with results')
    args = parser.parse_args()

    analyze_scalability_csv_raw(args.infile)

if __name__ == "__main__":
    main()
