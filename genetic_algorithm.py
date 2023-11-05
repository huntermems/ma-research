import numpy as np
import random
import config
import itertools
from hga_aco import HGAAntColonyOptimization

POPULATION_SIZE = 20
MAX_GENERATIONS = 300
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.6

# Local Search Parameters
NUMBER_OF_NEIGHBORS = 5
MAX_NO_IMPROVEMENT = 5


def objective_function(solution):
    return 1/config.total_t(solution)


def total_time(solution, reset_aisle=True):
    return config.total_t(solution, reset_aisle)


class HybridGeneticAlgorithm:

    previous_fitness = 0
    same_fitness_count = 0

    occurences = {}

    def __init__(self, random_instance):
        self.random_instance = random_instance

    # Genetic Algorithm Functions
    def create_individual(self):
        individual = []
        for item in config.ORDER:
            chosen_item = ''
            while True:
                item_locations = config.item_location_mapping[item]
                chosen_item = item_locations[np.random.choice(
                    len(item_locations))]
                if chosen_item not in individual:
                    break
            individual.append(chosen_item)
        self.random_instance.shuffle(individual)
        return individual

    def create_population(self):
        population = []
        for _ in range(POPULATION_SIZE):
            while True:
                individual = self.create_individual()
                if individual not in population:
                    population.append(individual)
                    break
        return population

    def evaluate_population(self, population):
        return [objective_function(solution) for solution in population]

    def select_parents(self, population, fitnesses, num_parents=2):
        parents = []
        copy_fitnesses = fitnesses.copy()
        i = 0
        while i < num_parents:
            max_value = max(copy_fitnesses)
            idx = copy_fitnesses.index(max_value)
            if i >= 1 and population[idx] in parents:
                copy_fitnesses[idx] = -99999
                continue
            parents.append(population[idx])
            copy_fitnesses[idx] = -99999
            i += 1

        return parents

    def crossover(self, parents):
        """Perform PMX crossover on two parent chromosomes"""
        if self.random_instance.random() > CROSSOVER_PROBABILITY:
            return None, None
        parent1 = parents[0]
        parent2 = parents[1]
        length = len(parent1)
        if parent1 == parent2:
            print(parents)
            exit()
        # Choose two random crossover points
        cxpoint1 = 0
        cxpoint2 = 0
        while cxpoint1 == cxpoint2 or (set([cxpoint1, cxpoint2]) == set([0, length - 1]) and length > 2):
            cxpoint1 = self.random_instance.randint(0, length - 1)
            cxpoint2 = self.random_instance.randint(0, length - 1)
        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Initialize offspring chromosomes
        child1 = [0 for _ in range(length)]
        child2 = [0 for _ in range(length)]

        # Iterate over the crossover points and perform the PMX
        for i in range(cxpoint1, cxpoint2 + 1):
            # Get the values at the current position in each parent
            value1 = parent1[i]
            value2 = parent2[i]

            # Swap the values in the offspring
            child1[i] = value2
            child2[i] = value1

        child1_types = [config.get_order_type(c) for c in child1]
        child2_types = [config.get_order_type(c) for c in child2]
        # Iterate over the rest of the values and perform the PMX
        for i in range(length):
            # Skip the values in the crossover range
            if i >= cxpoint1 and i <= cxpoint2:
                continue

            # Get the values at the current position in each parent
            value1 = parent1[i]
            idx1 = i
            value1_type = config.get_order_type(value1)
            # If the value is not already in the offspring, copy it over
            while value1 in child1 or child1_types.count(value1_type) == self.occurences[value1_type]:
                try:
                    idx1 = child1.index(value1)
                except Exception as e:
                    idx1 = child1_types.index(value1_type)
                value1 = child2[idx1]
                value1_type = config.get_order_type(value1)
            child1[i] = value1
            child1_types[i] = config.get_order_type(value1)

        for i in range(length):
            # Skip the values in the crossover range
            if i >= cxpoint1 and i <= cxpoint2:
                continue

            # Get the values at the current position in each parent
            value2 = parent2[i]
            idx2 = i
            value2_type = config.get_order_type(value2)
            # If the value is not already in the offspring, copy it over
            while value2 in child2 or child2_types.count(value2_type) == self.occurences[value2_type]:
                try:
                    idx2 = child2.index(value2)
                except Exception as e:
                    idx2 = child2_types.index(value2_type)
                value2 = child1[idx2]
                value2_type = config.get_order_type(value2)

            child2[i] = value2
            child2_types[i] = config.get_order_type(value2)

        if not all([len(set(child1)) == config.ORDER_LENGTH, len(set(child2)) == config.ORDER_LENGTH,
                    len(set(child1_types)) == config.ORDER_LENGTH, len(set(child2_types)) == config.ORDER_LENGTH]):
            print(cxpoint1, cxpoint2)
            print(len(set(parent1)), len(set(parent2)),
                  config.ORDER_LENGTH)
            print(f"Error Parent: {parent1} \n {parent2}")
            print(len(set(child1)), len(set(child2)),
                  config.ORDER_LENGTH)
            print(
                f"Error Parent Type: {[config.get_order_type(p) for p in parent1]} \n {[config.get_order_type(p) for p in parent2]}")
            print(f"Error Child Type: {child1_types} \n {child2_types}")
            config.exit(f"Error Child: {child1} \n {child2}")

        # print(length)
        # print(parents)
        # print(child1, child2)
        return child1, child2

    def mutate(self, individual):
        if self.random_instance.random() < MUTATION_PROBABILITY:
            # Randomly select the index of the gene
            indexes_of_gene_to_change = np.random.choice(len(individual))
            selected_gene = individual[indexes_of_gene_to_change]

            # Get the item name
            item_name = config.warehouse[selected_gene[0]
                                         ][selected_gene[1]][selected_gene[2]]

            # Get the item's list of location
            item_locations = config.item_location_mapping[item_name]

            item_new = selected_gene

            # Select a new location for the item
            while item_new in individual:
                item_new = item_locations[np.random.choice(
                    len(item_locations))]

            individual[indexes_of_gene_to_change] = item_new
            return individual
        else:
            return None

    # Local Search Functions

    def get_neighbors(self, individual):
        neighbors = []
        for _ in range(NUMBER_OF_NEIGHBORS):
            copy_of_individual = individual.copy()
            first_index = self.random_instance.randint(
                0, len(copy_of_individual)-1)
            second_index = self.random_instance.randint(
                0, len(copy_of_individual)-1)
            copy_of_individual[first_index], copy_of_individual[second_index] = copy_of_individual[second_index], copy_of_individual[first_index]
            neighbors.append(copy_of_individual)

        return neighbors

    def local_hga_search(self, initial_individual, probablity, max_no_improvement):
        current_individual = initial_individual
        if self.random_instance.random() < probablity:
            current_fitness = objective_function(current_individual)
            no_improvement = 0
            while no_improvement < max_no_improvement:
                neighbors = self.get_neighbors(current_individual)
                neighbor_fitnesses = [objective_function(
                    neighbor) for neighbor in neighbors]
                if len(neighbor_fitnesses):
                    best_neighbor_fitness = max(neighbor_fitnesses)
                    if current_fitness < best_neighbor_fitness:
                        best_neighbor_index = neighbor_fitnesses.index(
                            best_neighbor_fitness)
                        current_individual = neighbors[best_neighbor_index]
                        current_fitness = best_neighbor_fitness
                no_improvement += 1

        return current_individual

    def local_aco_search(self, initial_individual, probablity, max_no_improvement):
        current_individual = initial_individual
        if self.random_instance.random() < probablity:
            aco = HGAAntColonyOptimization(num_ants=10, num_iterations=5,
                                        pheromone_weight=1.0, heuristic_weight=2.0, evaporation_rate=0.1, num_cities=len(current_individual))
            for i in range(len(current_individual)):
                for j in range(len(current_individual)):
                    aco.add_distance(i, j, total_time(
                        [current_individual[i], current_individual[j]], reset_aisle=False))

            best_path, _ = aco.run()
            current_individual = [current_individual[b] for b in best_path]
            config.current_aisle_of_sr = config.INITIAL_SR_AISLE
        return current_individual

    def local_search(self, initial_individual, probablity, max_no_improvement, aco):
        if aco:
            return self.local_aco_search(initial_individual, probablity, max_no_improvement)
        else:
            return self.local_hga_search(initial_individual, probablity, max_no_improvement)

    def evolve_population(self, population, local_search_prob, aco):
        copy_population = population.copy()
        fitnesses = self.evaluate_population(copy_population)
        new_population = []
        for i in range(POPULATION_SIZE // 2):
            parents = self.select_parents(copy_population, fitnesses)
            offspring1, offspring2 = self.crossover(parents)
            # print("Iter ", i)
            # print("Parents ", parents)
            # print("Offspings ", offspring1, offspring2)
            if offspring1 == None and offspring2 == None:
                continue
            for offspring in [offspring1, offspring2]:
                mutation = self.mutate(offspring)
                if mutation is not None and mutation not in population and mutation not in new_population:
                    new_population.append(mutation)
                elif offspring not in population and offspring not in new_population:
                    new_population.append(offspring)
        new_population.extend(
            population[:POPULATION_SIZE - len(new_population)])
        for i in range(len(new_population)):
            new_population[i] = self.local_search(
                new_population[i], local_search_prob, MAX_NO_IMPROVEMENT, aco)

        return new_population

    def hga(self, local_search_prob, aco=False):
        best_solution = None
        best_fitness = 0
        best_time = 0
        best_generation = 0
        population = self.create_population()
        for item in config.ORDER:
            self.occurences[item] = config.ORDER.count(item)
        for generation in range(MAX_GENERATIONS):
            individual_order = []
            population = self.evolve_population(
                population, local_search_prob, aco)
            best_individual = max(
                population, key=lambda individual: objective_function(individual))
            for item in best_individual:
                individual_order.append(
                    config.warehouse[item[0]][item[1]][item[2]])
            fitness = objective_function(best_individual)
            current_time = config.total_t(best_individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = best_individual
                best_time = current_time
                best_generation = generation
            print(
                f"Generation {generation}: Best Solution = {individual_order}, Location = {best_individual}, Best Time = {current_time}")

        # print(population)
        # for p in population:
        #     id = []
        #     for item in p:
        #         id.append(config.get_order_type(item))
        #     print(id)
        if best_solution:
            solution_item_order = []
            for item in best_solution:
                solution_item_order.append(
                    config.warehouse[item[0]][item[1]][item[2]])
            # print(
            #     f"Best generation {best_generation}: {solution_item_order}, Location: {best_solution}, Time: {best_time}, Fitness = {best_fitness}")
        return best_time
