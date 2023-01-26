import random
from create_converged_job import create_run_script, boson_converged_job_id, ipi_args

def main():
	nbeads = 36
	temperature = 5.804681293240046 # kelvin

	num_repetitions = 3
	num_clients = 1

	random.seed(0)
	past_seeds = set()

	for num_bosons in [2,3,4,32,64]:
		for i in range(num_repetitions):
			job_id = boson_converged_job_id(num_bosons, nbeads, temperature, i)

			ipi_socket = 350000 + 100 * num_bosons + i

			seed = random.randint(1,1000000)
			assert seed not in past_seeds
			past_seeds.add(seed)

			ipi_runline = ipi_args(num_bosons, nbeads, temperature, seed, ipi_socket, num_clients)

			create_run_script(job_id, ipi_runline, ipi_socket)

if __name__ == "__main__":
    main()