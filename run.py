import math
import config
import datetime
import time
import random
import sourcerandom
import numpy as np
from genetic_algorithm import HybridGeneticAlgorithm
from aco import AntColonyOptimization
from pso import PSO
from enumeration import Enumeration

RAND_GEN = sourcerandom.SourceRandom(
    source=sourcerandom.OnlineRandomnessSource.QRNG_ANU)

NO_OF_TEST = 1


def test():
    run_ga = True
    run_hga = True
    run_hga_aco = True
    run_aco = True
    run_pso = True
    run_enum = True
    results = []
    f = open(f'result-{str(datetime.datetime.now())}.txt', 'w')
    for density in [0.6,0.75,0.9]:
        config.DENSITY = density
        for aisle in [1,2,3,4]:
            config.NUMBER_OF_AISLE = aisle
            if aisle == 4:
                run_enum = False
            else:
                run_enum = True
            if aisle > 2:
                config.ITEM_NUMERATION = config.BIG_ITEM_NUMERATION
                config.recompute()
            for shape in [1,0.73,0.6]:
                if shape == 1:
                    config.HEIGHT_OF_STORAGE_BIN = 3.4
                    config.LENGTH_OF_STORAGE_BIN = 15.5
                    config.T_H = config.LENGTH_OF_STORAGE_BIN*config.NUMBER_OF_COLUMN/config.HORIZONTAL_VELOCITY
                    config.T_V = config.HEIGHT_OF_STORAGE_BIN*config.NUMBER_OF_ROWS/config.VERTICAL_VELOCITY
                    config.BIG_T = max([ config.T_H, config.T_V ])
                    config.SHAPE_FACTOR = min([config.T_H / config.BIG_T, config.T_V / config.BIG_T])
                    config.TIME_RATIO = 1 + (np.power(config.SHAPE_FACTOR, 2))*1/3
                elif shape == 0.73:
                    config.HEIGHT_OF_STORAGE_BIN = 3.4
                    config.LENGTH_OF_STORAGE_BIN = 15.5
                    config.T_H = config.LENGTH_OF_STORAGE_BIN*config.NUMBER_OF_COLUMN/config.HORIZONTAL_VELOCITY
                    config.T_V = config.HEIGHT_OF_STORAGE_BIN*config.NUMBER_OF_ROWS/config.VERTICAL_VELOCITY
                    config.BIG_T = max([ config.T_H, config.T_V ])
                    config.SHAPE_FACTOR = min([config.T_H / config.BIG_T, config.T_V / config.BIG_T])
                    config.TIME_RATIO = 1 + (np.power(config.SHAPE_FACTOR, 2))*1/3
                else:
                    config.HEIGHT_OF_STORAGE_BIN = 126.6
                    config.LENGTH_OF_STORAGE_BIN = 16.7
                    config.T_H = config.LENGTH_OF_STORAGE_BIN*config.NUMBER_OF_COLUMN/config.HORIZONTAL_VELOCITY
                    config.T_V = config.HEIGHT_OF_STORAGE_BIN*config.NUMBER_OF_ROWS/config.VERTICAL_VELOCITY
                    config.BIG_T = max([ config.T_H, config.T_V ])
                    config.SHAPE_FACTOR = min([config.T_H / config.BIG_T, config.T_V / config.BIG_T])
                    config.TIME_RATIO = 1 + (np.power(config.SHAPE_FACTOR, 2))*1/3
                config.recompute()
                result = []
                if run_enum:
                    start_time_enum = time.perf_counter()
                    enum = Enumeration(config.item_location_mapping)
                    travel_time_with_enum = enum.run(config.ORDER)
                    executed_time_with_enum = time.perf_counter() - start_time_enum
                    result.append(travel_time_with_enum)
                    result.append(executed_time_with_enum)
                if run_ga:
                    for _ in range(NO_OF_TEST):
                        local_search_prob = 0
                        start_time_without_ls = time.perf_counter()
                        instance1 = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
                            RAND_GEN.randbytes(random.randint(5, 10))))
                        travel_time_wo_ls = instance1.hga(local_search_prob)
                        time_without_ls = time.perf_counter() - start_time_without_ls
                        result.append(travel_time_wo_ls)
                        result.append(time_without_ls)

                if run_hga:
                    for _ in range(NO_OF_TEST):
                        local_search_prob = 0.3 if config.ORDER_LENGTH > 1 else 0
                        start_time_with_ls = time.perf_counter()
                        instance = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
                            RAND_GEN.randbytes(random.randint(5, 10))))
                        travel_time_w_ls = instance.hga(local_search_prob)
                        time_with_ls = time.perf_counter() - start_time_with_ls
                        result.append(travel_time_w_ls)
                        result.append(time_with_ls)

                if run_hga_aco:
                    for _ in range(NO_OF_TEST):
                        local_search_prob = 0.2 if config.ORDER_LENGTH > 1 else 0
                        start_time_with_aco_ls = time.perf_counter()
                        instance = HybridGeneticAlgorithm(random_instance=random.SystemRandom(
                            RAND_GEN.randbytes(random.randint(5, 10))))
                        travel_time_w_aco_ls = instance.hga(local_search_prob, aco=True)
                        time_with_aco_ls = time.perf_counter() - start_time_with_aco_ls
                        result.append(travel_time_w_aco_ls)
                        result.append(time_with_aco_ls)

                if run_aco:
                    for _ in range(NO_OF_TEST):
                        start_time_aco = time.perf_counter()
                        aco = AntColonyOptimization(
                            numAnts=300,
                            numIterations=10,
                            pheromoneWeight=8,
                            heuristicWeight=3,
                            evaporationRate=0.1,
                            order=config.ORDER,
                            itemMapping=config.item_location_mapping,
                            randomInstance=random.SystemRandom(RAND_GEN.randbytes(random.randint(5, 10))))
                        bestPath, bestPathLength = aco.run()
                        executed_time_with_aco = time.perf_counter() - start_time_aco
                        result.append(bestPathLength)
                        result.append(executed_time_with_aco)

                if run_pso:
                    for _ in range(NO_OF_TEST):
                        start_time_pso = time.perf_counter()
                        pso = PSO(orderLength=config.ORDER_LENGTH,numParticles=min([math.factorial(config.ORDER_LENGTH), 20]))
                        bestPathPSO, bestTimePSO = pso.run()
                        executed_time_pso = time.perf_counter() - start_time_pso
                        result.append(bestTimePSO)
                        result.append(executed_time_pso)
                result = [round(n,2) for n in result]
                result = [str(n) for n in result]
                state = f"Number of item: {config.ORDER_LENGTH} Aisles no: {aisle}, Shape: {shape}, Density: {density}, {','.join(result)}\n"
                f.write(state)
                f.flush()
    f.close()
    return results

test()

      

                

# if run_ga:
#     print("Travel time GA: ", travel_time_wo_ls_list)
# if run_hga:
#     print("Travel time HGA: ", travel_time_w_ls_list)
# if run_hga_aco:
#     print("Travel time HGA-ACO: ", travel_time_w_aco_ls_list)
# if run_aco:
#     print("Travel time with original aco search: ", travel_time_w_aco)
# if run_pso:
#     print("Travel time with pso search: ", travel_time_pso)
# if run_enum:
#     print("Travel time with enum: ", travel_time_with_enum)

# if run_ga:
#     print("Time executed w/o local search: ", average_executed_time_wo_ls)
# if run_hga:
#     print("Time executed with local search: ", average_executed_time_w_ls)
# if run_hga_aco:
#     print("Time executed with aco local search: ",
#           average_executed_time_w_aco_ls)
# if run_aco:
#     print("Time executed with original aco search: ",
#           average_executed_time_w_aco)
# if run_pso:
#     print("Time executed with pso search: ",
#           average_executed_time_pso)
# if run_enum:
#     print("Time executed with enum: ", executed_time_with_enum)

# if run_enum:
#     print("Average Time executed with enum: ", executed_time_with_enum)
# if run_ga:
#     print("Average Time executed w/o local search: ",
#           sum(average_executed_time_wo_ls)/NO_OF_TEST)
# if run_hga:
#     print("Average Time executed with local search: ",
#           sum(average_executed_time_w_ls)/NO_OF_TEST)
# if run_hga_aco:
#     print("Average Time executed with aco local search: ",
#           sum(average_executed_time_w_aco_ls)/NO_OF_TEST)
# if run_aco:
#     print("Average Time executed with original aco: ",
#           sum(average_executed_time_w_aco)/NO_OF_TEST)
# if run_pso:
#     print("Average Time executed with pso: ",
#           sum(average_executed_time_pso)/NO_OF_TEST)
