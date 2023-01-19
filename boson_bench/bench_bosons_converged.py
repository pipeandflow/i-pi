import time
import sys
import logging
import subprocess
import re
import statistics
# import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import itertools
import tempfile
from scipy import stats
import math
import os
import argparse

BOSON_SCALING_CSV_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "boson_scaling.csv")
NUM_REPETITIONS = 1

global args

def ipi_config(boson_positions, boson_masses, boson_labels, bosons_list,
              nbeads, temperature, seed, ipi_socket):
    INPUT_XML_TEMPLATE = """
<simulation threading='False' verbosity='low'>

<ffsocket mode='unix' name='driver'>
        <address> {ipi_socket}  </address>
</ffsocket>

<total_steps> 1000000 </total_steps>

<output prefix="data">
  <trajectory stride="100" filename="pos" cell_units="angstrom">positions{{angstrom}}</trajectory>
  <!--<trajectory stride="1" filename="xc" format="xyz">x_centroid{{angstrom}}</trajectory>-->
  <properties stride="100"> [ step, time{{femtosecond}}, conserved, temperature{{kelvin}}, kinetic_cv,
        potential, virial_fq ] </properties>
</output>

<prng>
  <seed> {seed} </seed>
</prng>

<system>

  <forces>
      <force forcefield="driver"></force>
  </forces>

  <initialize nbeads="{nbeads}">
    <positions mode="manual" bead="0"> {boson_positions} </positions>
    <masses mode="manual"> {boson_masses} </masses>
    <labels mode="manual"> {boson_labels} </labels>
    <cell>
     [   2500, 0, 0, 0, 2500, 0, 0, 0, 2500 ]
    </cell>
    <velocities mode='thermal' units='kelvin'> {temperature} </velocities>
  </initialize>

  <normal_modes propagator='bab'>
          <nmts> 10 </nmts>
          <bosons> {bosons_list} </bosons>
  </normal_modes>

  <ensemble>
     <temperature units="kelvin"> {temperature} </temperature>
  </ensemble>

  <motion mode="dynamics">
    <fixcom> False </fixcom>
    <dynamics mode="nvt">
     <timestep units="femtosecond"> 1 </timestep>
      <thermostat mode='pile_l'>
            <tau units='femtosecond'>100</tau>
      </thermostat>

    </dynamics>
  </motion>

</system>

</simulation>
"""
    return INPUT_XML_TEMPLATE.format(boson_positions=boson_positions, boson_masses=boson_masses, boson_labels=boson_labels, bosons_list=bosons_list,
                                    temperature=temperature, nbeads=nbeads, seed=seed,
                                    ipi_socket=ipi_socket)

global logger

def set_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[
                                            logging.FileHandler("%s-bench.log" % time.strftime("%Y%m%d-%H%M%S")),
                                            logging.StreamHandler()
                                    ])

    global logger
    logger = logging.getLogger(__name__)

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def stringify_elementwise(lst):
    return [str(x) for x in lst]

def time_ipi(input_filename):
    global args

    # Run i-pi

    # duplicated from ipi/ipi_tests/test_tools.py: Runner.run()

    call_ipi="i-pi " + input_filename

    clientcall = "i-pi-driver -u -m harm3d -o 1.21647924E-8 -h " + str(args.ipi_socket)

    logger.debug("Start i-pi server")

    ipi = subprocess.Popen(
        call_ipi,
        # cwd=(self.tmp_dir),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # if len(clients) > 0:
    #     f_connected = False
    #     for client in clients:
    #         for i in range(50):
    #             if os.path.exists("/tmp/ipi_" + client[2]):
    #                 f_connected = True
    #                 break
    #             else:
    #                 time.sleep(0.5)
    #         if not f_connected:
    #             print("Could not find the i-PI UNIX socket.")
    #             return "Could not find the i-PI UNIX socket"

    # Run drivers by defining cmd2 which will be called, eventually
    driver = list()

    # for client in clients:
    #     if client[1] == "unix":
    #         clientcall = call_driver + " -m {} {} {} -u ".format(
    #             client[0], address_key, client[2]
    #         )
    #     elif client[1] == "inet":
    #         clientcall = call_driver + " -m {} {} {} -p {}".format(
    #             client[0], client[2], address_key, client[3]
    #         )

    #     else:
    #         raise ValueError("Driver mode has to be either unix or inet")

    cmd = clientcall

    # Add extra flags if necessary
    # if any("-" in str(s) for s in client):
    #     flag_indeces = [
    #         i for i, elem in enumerate(client) if "-" in str(elem)
    #     ]
    #     for i, ll in enumerate(flag_indeces):
    #         if i < len(flag_indeces) - 1:
    #             cmd += " {} {}".format(
    #                 client[ll],
    #                 ",".join(client[ll + 1 : flag_indeces[i + 1]][:]),
    #             )
    #         else:
    #             cmd += " {} {}".format(
    #                 client[ll], ",".join(client[ll + 1 :][:])
    #             )
    # print("cmd:", cmd)

    time.sleep(5)
    logger.debug("Start driver execution")


    start_time = time.time()

    for _ in range(0, args.num_clients):
        driver.append(
            subprocess.Popen(cmd,
                             # cwd=(cwd),
                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        )

    logger.debug("Start wait for ipi process")

    ipi_error = ipi.communicate()[1].decode("ascii")
    if ipi_error != "":
        print(ipi_error)
    assert "" == ipi_error, "IPI ERROR OCCURRED: {}".format(ipi_error)

    end_time = time.time()

    logger.debug("i-pi run ended; time %s" % str(end_time - start_time))

    return end_time - start_time

def bench_ipi(input_xml_str):
        # TODO: store output

        # code duplicated from ipi/ipi_tests/test_tools.py: Runner.run()
    with tempfile.NamedTemporaryFile(mode="wt") as tmp:
        tmp.write(input_xml_str)
        tmp.flush()
        measured_time = time_ipi(tmp.name)
        return measured_time

def bench_bosons_single(nbosons, boson_positions):
    global args

    assert len(boson_positions) == nbosons

    boson_positions_config = str(flatten(boson_positions))

    boson_labels = str(["E"] * nbosons)
    boson_masses = str(["1.0"] * nbosons)
    bosons_list = str(list(range(nbosons)))

    config = ipi_config(boson_positions_config, boson_masses, boson_labels, bosons_list,
                        args.nbeads, args.temperature, args.seed,
                        args.ipi_socket)
    logger.debug("ipi config: %s" % config)
    measured_time = bench_ipi(config)
    logger.info("time measurement. nbosons: %d; time: %f" % (nbosons, measured_time))
    return measured_time

def random_position():
    RANDOM_RANGE_LOWER = -100
    RANDOM_RANGE_UPPER = 100
    return random.randint(RANDOM_RANGE_LOWER, RANDOM_RANGE_UPPER)

def random_boson_positions(nbosons):
    return [[random_position(), random_position(), random_position()] for _ in range(nbosons)]

def bench_bosons(nbosons):
    boson_positions = random_boson_positions(nbosons)

    time_measurements = [bench_bosons_single(nbosons, boson_positions) for _ in range(0, NUM_REPETITIONS)]
    logger.info("standard deviation. nbosons: %d; time: %f" % (nbosons, statistics.stdev(time_measurements)))
    return statistics.mean(time_measurements)

def boson_scalability(boson_numbers):
    with open(BOSON_SCALING_CSV_OUTPUT_PATH, "wt") as csv_log:
        w = csv.DictWriter(csv_log, ["nbosons", "time"])
        w.writeheader()
        csv_log.flush()

        for nbosons in boson_numbers:
            time = bench_bosons(nbosons)
            w.writerow({"nbosons": nbosons, "time": time})
            csv_log.flush()

def main():
    set_logger()

    # e.g. python bench_bosons_converged.py 3 32 17.4 1885 31415 -j 32
    parser = argparse.ArgumentParser(description='10^6 steps of PIMD-B')
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
    global args
    args = parser.parse_args()
    logger.debug("args: %s" % str(args))

    logger.debug("python seed: %d" % args.seed) # same as i-pi seed
    random.seed(args.seed)

    boson_scalability([args.num_bosons])

if __name__ == "__main__":
    main()
