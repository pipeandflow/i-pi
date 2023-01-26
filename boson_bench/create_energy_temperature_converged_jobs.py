import random
from create_converged_job import create_run_script, boson_converged_job_id, ipi_args
import numpy as np

def temperature_for_hbw(hbw):
	# returns temperature in Kelvin
	spring_constant = 1.21647924E-8
	mass = 1.0
	omega = np.sqrt(spring_constant / mass)
	hbar = 1.0
	Boltzmann = 1.0

	t_hartree = (hbar * omega) / (hbw * Boltzmann)
	return t_hartree / 3.1668152e-06

def main():
	nbeads = 36

	num_bosons = 16

	num_repetitions = 3
	num_clients = 4

	random.seed(0)
	past_seeds = set()

	for hbw in [2, 3, 4, 5, 6]:
		temperature = temperature_for_hbw(hbw)

		for i in range(num_repetitions):
			job_id = boson_converged_job_id(num_bosons, nbeads, temperature, i)

			ipi_socket = 450000 + 100 * hbw + i

			seed = random.randint(1,1000000)
			assert seed not in past_seeds
			past_seeds.add(seed)

			ipi_runline = ipi_args(num_bosons, nbeads, temperature, seed, ipi_socket, num_clients)

			create_run_script(job_id, ipi_runline, ipi_socket)

if __name__ == "__main__":
    main()