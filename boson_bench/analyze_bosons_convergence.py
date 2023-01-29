import re
import numpy as np
import statistics
import argparse

def read_ipi_output(filename):
    """ Reads an i-PI output file and returns a dictionary with the properties in a tidy order. """
    
    f = open(filename, "r")
    
    regex = re.compile(".*column *([0-9]*) *--> ([^ {]*)")
    
    fields = []; cols = []
    for line in f:
        if line[0] == "#":
            match = regex.match(line)
            if match is None:
                print("Malformed comment line: ", line)
                continue # TODO: was error
            fields.append(match.group(2))
            cols.append(slice(int(match.group(1))-1,int(match.group(1))))
        else:
            break # done with header
    f.close()
    
    columns = {}
    raw = np.loadtxt(filename)
    for i, c in enumerate(fields):
        columns[c] = raw[:,cols[i]].T
        if columns[c].shape[0] == 1:
            columns[c].shape = columns[c].shape[1]
    return columns

def analyze_convergence_data_out(infile):
    o = read_ipi_output(infile)
    avg_kinetic = -statistics.mean(o['virial_fq'][500:])
    avg_potential = statistics.mean(o['potential'][500:])
    print("total number of points:", len(o['potential']))
    print("avg kinetic:", avg_kinetic)
    print("avg potential:", avg_potential)
    print("total:", avg_kinetic + avg_potential)

def main():
    parser = argparse.ArgumentParser(description='analyze boson convergence results')

    parser.add_argument('infile', type=str, nargs='*',
                        help='path to csv with results')
    args = parser.parse_args()

    infiles = args.infile

    for infile in args.infile:
        analyze_convergence_data_out(infile)

if __name__ == "__main__":
    main()