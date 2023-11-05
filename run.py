import config
import time
import random
import sourcerandom
from genetic_algorithm import HybridGeneticAlgorithm
from aco import AntColonyOptimization
from enumeration import Enumeration

RAND_GEN = sourcerandom.SourceRandom(
    source=sourcerandom.OnlineRandomnessSource.QRNG_ANU)

NO_OF_TEST = 1

run_ga = True
run_hga = True
run_hga_aco = False
run_aco = True
run_enum = True

if run_ga:
    average_executed_time_wo_ls = []
    travel_time_wo_ls_list = []
    for _ in range(NO_OF_TEST):
        local_search_prob = 0
        start_time_without_ls = time.perf_counter()
        instance1 = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
            RAND_GEN.randbytes(random.randint(5, 10))))
        travel_time_wo_ls = instance1.hga(local_search_prob)
        time_without_ls = time.perf_counter() - start_time_without_ls
        average_executed_time_wo_ls.append(time_without_ls)
        travel_time_wo_ls_list.append(travel_time_wo_ls)

if run_hga:
    average_executed_time_w_ls = []
    travel_time_w_ls_list = []
    for _ in range(NO_OF_TEST):
        local_search_prob = 0.5
        start_time_with_ls = time.perf_counter()
        instance = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
            RAND_GEN.randbytes(random.randint(5, 10))))
        travel_time_w_ls = instance.hga(local_search_prob)
        time_with_ls = time.perf_counter() - start_time_with_ls
        average_executed_time_w_ls.append(time_with_ls)
        travel_time_w_ls_list.append(travel_time_w_ls)

if run_hga_aco:
    average_executed_time_w_aco_ls = []
    travel_time_w_aco_ls_list = []
    for _ in range(NO_OF_TEST):
        local_search_prob = 0.2
        start_time_with_aco_ls = time.perf_counter()
        instance = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
            RAND_GEN.randbytes(random.randint(5, 10))))
        travel_time_w_aco_ls = instance.hga(local_search_prob, aco=True)
        time_with_aco_ls = time.perf_counter() - start_time_with_aco_ls
        average_executed_time_w_aco_ls.append(time_with_aco_ls)
        travel_time_w_aco_ls_list.append(travel_time_w_aco_ls)

if run_aco:
    average_executed_time_w_aco = []
    travel_time_w_aco = []
    for _ in range(NO_OF_TEST):
        start_time_aco = time.perf_counter()
        aco = AntColonyOptimization(50, 20, 1, 2, 0.2, config.ORDER, config.item_location_mapping, randonInstance=random.SystemRandom(
            RAND_GEN.randbytes(random.randint(5, 10))))
        bestPath, bestPathLength = aco.run()
        executed_time_with_aco = time.perf_counter() - start_time_aco
        average_executed_time_w_aco.append(executed_time_with_aco)
        travel_time_w_aco.append(bestPathLength)

if run_enum:
    start_time_enum = time.perf_counter()
    enum = Enumeration(config.item_location_mapping)
    travel_time_with_enum = enum.run(config.ORDER)
    executed_time_with_enum = time.perf_counter() - start_time_enum

if run_ga:
    print("Travel time w/o local search: ", travel_time_wo_ls_list)
if run_hga:
    print("Travel time with local search: ", travel_time_w_ls_list)
if run_hga_aco:
    print("Travel time with aco local search: ", travel_time_w_aco_ls_list)
if run_aco:
    print("Travel time with original aco search: ", travel_time_w_aco)
if run_enum:
    print("Travel time with enum: ", travel_time_with_enum)

if run_ga:
    print("Time executed w/o local search: ", average_executed_time_wo_ls)
if run_hga:
    print("Time executed with local search: ", average_executed_time_w_ls)
if run_hga_aco:
    print("Time executed with aco local search: ",
          average_executed_time_w_aco_ls)
if run_aco:
    print("Time executed with original aco search: ",
          average_executed_time_w_aco)
if run_enum:
    print("Time executed with enum: ", executed_time_with_enum)

if run_ga:
    print("Average Time executed w/o local search: ",
          sum(average_executed_time_wo_ls)/NO_OF_TEST)
if run_hga:
    print("Average Time executed with local search: ",
          sum(average_executed_time_w_ls)/NO_OF_TEST)
if run_hga_aco:
    print("Average Time executed with aco local search: ",
          sum(average_executed_time_w_aco_ls)/NO_OF_TEST)
if run_aco:
    print("Average Time executed with aco local search: ",
          sum(average_executed_time_w_aco)/NO_OF_TEST)
