import argparse
import os
from datetime import datetime
import pathlib
import stat

IPI_ROOT="/hirshblab/data/yotamfe1/mi-pi"
# IPI_ROOT="~/mi-pi"
STORAGE_ROOT_DIR="/hirshblab-storage/yotamfe1/"
# STORAGE_ROOT_DIR="test_storage"

RUN_SCRIPT_CONTENT_TEMPLATE="""
#!/bin/bash

DATE=$(date +%Y-%m-%d);
RESULTS_FOR_DATE_DIR={output_dir_root}/$DATE
RESULTS_DIR=$RESULTS_FOR_DATE_DIR/{job_id}

set -e

mkdir -p $RESULTS_FOR_DATE_DIR
mkdir $RESULTS_DIR

source {ipi_root}/env.sh
cp {ipi_root}/boson_bench/bench_bosons_converged.py $RESULTS_DIR/bench_bosons_converged.py
module load python/python-anaconda_3.7
cd $RESULTS_DIR

rm -f /tmp/ipi_{ipi_socket}
python3 $RESULTS_DIR/bench_bosons_converged.py {args}"""

def boson_converged_job_id(num_bosons, nbeads, temperature, details):
    job_id = "bosons_converged_%d_%d_%s" % (num_bosons, nbeads, str(temperature))
    if details is not None:
        job_id += "_%s" % details
    return job_id

def ipi_args(num_bosons, nbeads, temperature, seed, ipi_socket, num_clients):
    return "%d %d %f %d %d -j %d" % (num_bosons, nbeads, temperature, seed, ipi_socket, num_clients)

def create_run_script(job_id, ipi_args, ipi_socket):
    output_run_script_path = "./run_%s.sh" % job_id
    with open(output_run_script_path, "wt") as output_run_script:
        output_run_script.write(RUN_SCRIPT_CONTENT_TEMPLATE.format(ipi_root=IPI_ROOT, 
                                                                   output_dir_root=STORAGE_ROOT_DIR,
                                                                   job_id=job_id,
                                                                   ipi_socket=ipi_socket,
                                                                   args=ipi_args))

    f = pathlib.Path(output_run_script_path)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)

def main():
    parser = argparse.ArgumentParser(description='generate boson bench converged run script')
    parser.add_argument('num_bosons', metavar='N', type=int,
                        help='number of bosons')
    parser.add_argument('nbeads', metavar='P', type=int,
                        help='number of beads')
    parser.add_argument('temperature', metavar='T', type=float,
                        help='temperature')
    parser.add_argument('seed', type=int,
                        help='seed')
    parser.add_argument('ipi_socket', metavar='socket', type=int,
                        help='ipi internal socket')
    parser.add_argument('-j', '--num-clients', type=int, default=1,
                        help='number of force-field clients')

    parser.add_argument('-d', '--details', type=str, default=None,
                        help="extra identification of the job")

    args = parser.parse_args()

    job_id = boson_converged_job_id(args.num_bosons, args.nbeads, args.temperature, args.details)

    # pathlib.Path(output_dir).mkdir(parents=False, exist_ok=True) 

    ipi_runline = ipi_args(args.num_bosons, args.nbeads, args.temperature, args.seed, args.ipi_socket, args.num_clients)

    create_run_script(job_id, ipi_args, args.ipi_socket)
    
if __name__ == "__main__":
    main()
