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

    lognbosons = np.log2(np.asarray(nbosons))
    mean_time_per_boson = np.mean(times_per_nbosons, axis=1)
    mean_time_error = np.std(times_per_nbosons, axis=1)
    log_mean_time_per_boson = np.log2(mean_time_per_boson)

    x = lognbosons
    y = log_mean_time_per_boson
    err = mean_time_error / mean_time_per_boson
    print("std:", err)

    slope, intercept, r, p, std_err = stats.linregress(lognbosons, log_mean_time_per_boson)
    print(slope, intercept, r, p)

    plt.errorbar(x, y, err, linestyle='None', marker='o', ecolor='red')
    plt.plot(x, intercept + slope * x, '-', label=f'slope ${slope:.3f}$')

def main():
    parser = argparse.ArgumentParser(description='analyze boson scalability results')

    parser.add_argument('infile', type=str, nargs='*',
                        help='path to csv with results')
    args = parser.parse_args()

    for infile in args.infile:
        analyze_scalability_csv(infile)

    # plt.rcParams['font.size'] = 15
    # plt.rc('axes',linewidth=2,labelpad=10)
    # plt.rcParams["xtick.direction"] = "in"
    # plt.rcParams["ytick.direction"] = "in"
    # plt.rc('xtick.major',size=10, width=2)
    # plt.rc('xtick.minor',size=7, width=2)
    # plt.rc('ytick.major',size=10, width=2)
    # plt.rc('ytick.minor',size=7, width=2)

    plt.xlabel('N')
    plt.ylabel('time (s)')
    plt.gca().xaxis.set_major_formatter(lambda x, pos: str(int(pow(2, x))))
    plt.gca().yaxis.set_major_formatter(lambda y, pos: str(int(pow(2, y))))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
