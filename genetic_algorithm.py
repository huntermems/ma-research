import numpy as np
import random
import config
import time
from ant_colony_algorithm import AntColonyOptimization

POPULATION_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 1

MUTATION_UPPER_THRESHOLD = 0.6
MUTATION_LOWER_THRESHOLD = 0.4
MUTATION_RATE_STEP = 0.05

# Local Search Parameters
NUMBER_OF_NEIGHBORS = 1
MAX_NO_IMPROVEMENT = 1


def objective_function(solution):
    return 1/config.total_t(solution)


def total_time(solution, reset_aisle=True):
    return config.total_t(solution, reset_aisle)


class HybridGeneticAlgorithm:

    previous_fitness = 0
    same_fitness_count = 0

    def __init__(self, random_instance):
        self.random_instance = random_instance

    # Genetic Algorithm Functions
    def create_individual(self):
        individual = []
        for item in config.ITEM_NUMERATION:
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
        return [self.create_individual() for _ in range(POPULATION_SIZE)]

    def evaluate_population(self, population):
        return [objective_function(solution) for solution in population]

    def select_parents(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        parents = self.random_instance.choices(population, probabilities, k=2)
        return parents

    def crossover(self, parents):
        """Perform PMX crossover on two parent chromosomes"""
        if self.random_instance.random() > CROSSOVER_PROBABILITY:
            return None, None
        parent1 = parents[0]
        parent2 = parents[1]
        length = len(parent1)
        # Choose two random crossover points
        cxpoint1 = 0
        cxpoint2 = 0
        while cxpoint1 == cxpoint2:
            cxpoint1 = self.random_instance.randint(0, len(parent1) - 1)
            cxpoint2 = self.random_instance.randint(0, len(parent1) - 1)
        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Initialize offspring chromosomes
        child1 = parent2[:]
        child2 = parent1[:]

        # Iterate over the crossover points and perform the PMX
        for i in range(cxpoint1, cxpoint2 + 1):
            # Get the values at the current position in each parent
            value1 = parent1[i]
            value2 = parent2[i]

            # Swap the values in the offspring
            child1[i] = value1
            child2[i] = value2

        # Replace values outside the crossover range with 0
        for i in range(length):
            # Skip the values in the crossover range
            if i >= cxpoint1 and i <= cxpoint2:
                continue

            child1[i] = 0
            child2[i] = 0
        # print(cxpoint1, cxpoint2)
        # print(child1, child2)
        # print(length)

        # Iterate over the rest of the values and perform the PMX
        for i in range(length):
            # Skip the values in the crossover range
            if i >= cxpoint1 and i <= cxpoint2:
                continue

            # Get the values at the current position in each parent
            value1 = parent1[i]
            value2 = parent2[i]

            # If the value is not already in the offspring, copy it over
            if value1 not in child1:
                child1[i] = value1
            elif value2 not in child1:
                child1[i] = value2

            if value2 not in child2:
                child2[i] = value2
            elif value1 not in child2:
                child2[i] = value1

        if not all([len(set(child1)) == len(config.ITEM_NUMERATION), len(set(child2)) == len(config.ITEM_NUMERATION)]):
            print(cxpoint1, cxpoint2)
            print(len(set(parent1)), len(set(parent2)),
                  len(config.ITEM_NUMERATION))
            print(f"Error Parent: {parent1} \n {parent2}")
            print(len(set(child1)), len(set(child2)),
                  len(config.ITEM_NUMERATION))
            config.exit(f"Error Child: {child1} \n {child2}")

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
            first_index = self.random_instance.randint(0, len(copy_of_individual)-1)
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
            aco = AntColonyOptimization(num_ants=5, num_iterations=10,
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
        global MUTATION_PROBABILITY

        fitnesses = self.evaluate_population(population)
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parents = self.select_parents(population, fitnesses)
            offspring1, offspring2 = self.crossover(parents)
            if offspring1 == None and offspring2 == None:
                continue
            for offspring in [offspring1, offspring2]:
                mutation = self.mutate(offspring)
                if mutation is not None:
                    new_population.append(mutation)
                else:
                    new_population.append(offspring)
        new_population.extend(
            population[:POPULATION_SIZE - len(new_population)])
        for i in range(len(new_population)):
            new_population[i] = self.local_search(
                new_population[i], local_search_prob, MAX_NO_IMPROVEMENT, aco)

        # Increase mutation chance if same fitness score is repeated
        best_individual = max(
            new_population, key=lambda individual: objective_function(individual))
        current_fitness = objective_function(best_individual)
        if current_fitness == self.previous_fitness:
            if self.same_fitness_count > 10:
                if MUTATION_PROBABILITY < MUTATION_UPPER_THRESHOLD:
                    MUTATION_PROBABILITY += MUTATION_RATE_STEP
                self.same_fitness_count = 0
            self.same_fitness_count += 1
            self.previous_fitness = current_fitness
        else:
            if MUTATION_PROBABILITY > MUTATION_LOWER_THRESHOLD:
                MUTATION_PROBABILITY -= MUTATION_RATE_STEP
            self.previous_fitness = current_fitness
            self.same_fitness_count = 0
        return new_population

    def hga(self, local_search_prob, aco=False):
        best_solution = None
        best_fitness = 0
        best_time = 0
        best_generation = 0
        population = self.create_population()
        for generation in range(MAX_GENERATIONS):
            individual_order = []
            population = self.evolve_population(population, local_search_prob, aco)
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

        if best_solution:
            solution_item_order = []
            for item in best_solution:
                solution_item_order.append(
                    config.warehouse[item[0]][item[1]][item[2]])
            print(
                f"Best generation {best_generation}: {solution_item_order}, Location: {best_solution}, Time: {best_time}, Fitness = {best_fitness}")
        return best_generation
