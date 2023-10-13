import time
import random
import sourcerandom
from genetic_algorithm import HybridGeneticAlgorithm

RAND_GEN = sourcerandom.SourceRandom(source=sourcerandom.OnlineRandomnessSource.QRNG_ANU)

RAND_GEN.randbytes(random.randint(1,30))

NO_OF_TEST = 10

run_ga = False
run_hga = False
run_aco = True

if run_ga:
    average_generation_without_local_search = 0
    average_time_wo_ls = []
    generation_wo_ls = []
    for _ in range(NO_OF_TEST):
        local_search_prob = 0
        start_time_without_ls = time.perf_counter()
        instance1 = HybridGeneticAlgorithm(random_instance = random.SystemRandom(RAND_GEN.randbytes(random.randint(5,10))))
        generation_without_ls = instance1.hga(local_search_prob)
        average_generation_without_local_search += generation_without_ls
        time_without_ls = time.perf_counter() - start_time_without_ls
        average_time_wo_ls.append(time_without_ls)
        generation_wo_ls.append(generation_without_ls)

if run_hga:
    average_time_w_ls = []
    generation_w_ls = []
    average_generation_with_local_search = 0
    for _ in range(NO_OF_TEST):
        local_search_prob = 0.2
        start_time_with_ls = time.perf_counter()
        instance = HybridGeneticAlgorithm(random_instance = random.SystemRandom(RAND_GEN.randbytes(random.randint(5,10))))
        generation_with_ls = instance.hga(local_search_prob)
        average_generation_with_local_search += generation_with_ls
        time_with_ls = time.perf_counter() - start_time_with_ls
        average_time_w_ls.append(time_with_ls)
        generation_w_ls.append(generation_with_ls)

if run_aco:
    average_time_w_aco_ls = []
    generation_w_aco_ls = []
    average_generation_with_aco_local_search = 0
    for _ in range(NO_OF_TEST):
        local_search_prob = 0.03
        start_time_with_aco_ls = time.perf_counter()
        instance = HybridGeneticAlgorithm(random_instance = random.SystemRandom(RAND_GEN.randbytes(random.randint(5,10))))
        generation_with_aco_ls = instance.hga(local_search_prob, aco=True)
        average_generation_with_aco_local_search += generation_with_aco_ls
        time_with_aco_ls = time.perf_counter() - start_time_with_aco_ls
        average_time_w_aco_ls.append(time_with_aco_ls)
        generation_w_aco_ls.append(generation_with_aco_ls)

if run_ga:
    print("Generation w/o local search: ", generation_wo_ls)
if run_hga:
    print("Generation with local search: ", generation_w_ls)
if run_aco:
    print("Generation with aco local search: ", generation_w_aco_ls)

if run_ga:
    print("Average generation that produces best result (w/o ls): ",
    average_generation_without_local_search / NO_OF_TEST)
if run_hga:
    print("Average generation that produces best result (with ls): ",
    average_generation_with_local_search / NO_OF_TEST)
if run_aco:
    print("Average generation that produces best result (with aco ls): ",
    average_generation_with_aco_local_search / NO_OF_TEST)

if run_ga:
    print("Time executed w/o local search: ", average_time_wo_ls)
if run_hga:
    print("Time executed with local search: ", average_time_w_ls)
if run_aco:
    print("Time executed with aco local search: ", average_time_w_aco_ls)

if run_ga:
    print("Average Time executed w/o local search: ", sum(average_time_wo_ls)/NO_OF_TEST)
if run_hga:
    print("Average Time executed with local search: ", sum(average_time_w_ls)/NO_OF_TEST)
if run_aco:
    print("Average Time executed with aco local search: ", sum(average_time_w_aco_ls)/NO_OF_TEST)
