import numpy as np
import random
import config

POPULATION_SIZE = 50
MAX_GENERATIONS = 500
MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.4

# Local Search Parameters
MAX_NO_IMPROVEMENT = 10

random_instance = random.SystemRandom()

def objective_function(solution):
    return 1/config.total_t(solution)

# Genetic Algorithm Functions
def create_individual():
    individual = []
    for item in config.ITEM_NUMERATION:
        chosen_item = ''
        while True: 
            item_locations = config.item_location_mapping[item]
            chosen_item = item_locations[np.random.choice(len(item_locations))]
            if chosen_item not in individual:
                break
        individual.append(chosen_item)
    np.random.shuffle(individual)
    return individual

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    return [objective_function(solution) for solution in population]

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    parents = random.choices(population, probabilities, k=2)
    return parents

def crossover(parents):
    """Perform PMX crossover on two parent chromosomes"""
    if random_instance.random() > CROSSOVER_PROBABILITY:
        return None, None
    parent1 = parents[0]
    parent2 = parents[1]
    length = len(parent1)
    # Choose two random crossover points
    cxpoint1 = np.random.randint(0, len(parent1) - 1)
    cxpoint2 = np.random.randint(0, len(parent1) - 1)
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
        child1[child1.index(value2)] = value1
        child2[child2.index(value1)] = value2
        
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
        if value2 not in child2:
            child2[i] = value2            
    return child1, child2


def mutate(individual):
    if random_instance.random() < MUTATION_PROBABILITY:
        # Randomly select the index of the gene
        indexes_of_gene_to_change = np.random.choice(len(individual))
        selected_gene = individual[indexes_of_gene_to_change]

        # Get the item name
        item_name = config.warehouse[selected_gene[0]][selected_gene[1]][selected_gene[2]]

        # Get the item's list of location
        item_locations = config.item_location_mapping[item_name]

        item_new = indexes_of_gene_to_change

        # Select a new location for the item
        while item_new == indexes_of_gene_to_change:
            item_new = item_locations[np.random.choice(len(item_locations))]
            
        individual[indexes_of_gene_to_change] = item_new
        return individual
    else:
        return None

# Local Search Functions
def get_neighbors(individual):
    x, y = individual
    neighbors = []
    for dx in [-0.5, 0, 0.5]:
        for dy in [-0.5, 0, 0.5]:
            neighbor = (x + dx, y + dy)
            if neighbor != individual and -10 <= neighbor[0] <= 10 and -10 <= neighbor[1] <= 10:
                neighbors.append(neighbor)
    return neighbors

def local_search(initial_individual):
    current_individual = initial_individual
    current_fitness = objective_function(current_individual)
    no_improvement = 0
    while no_improvement < MAX_NO_IMPROVEMENT:
        neighbors = get_neighbors(current_individual)
        neighbor_fitnesses = [objective_function(neighbor) for neighbor in neighbors]
        if len(neighbor_fitnesses):
            best_neighbor_fitness = min(neighbor_fitnesses)
            if best_neighbor_fitness < current_fitness:
                best_neighbor_index = neighbor_fitnesses.index(best_neighbor_fitness)
                current_individual = neighbors[best_neighbor_index]
                current_fitness = best_neighbor_fitness
        no_improvement += 1
    return current_individual


def evolve_population(population):
    fitnesses = evaluate_population(population)
    new_population = []
    for _ in range(POPULATION_SIZE // 2):
        parents = select_parents(population, fitnesses)
        offspring1, offspring2 = crossover(parents)
        if offspring1 == None and offspring2 == None:
            continue
        for offspring in [offspring1, offspring2]:
            mutation = mutate(offspring)
            if mutation is not None:
                new_population.append(mutation)
            else:
                new_population.append(offspring)
    new_population.extend(population[:POPULATION_SIZE - len(new_population)])
    # for i in range(len(new_population)):
    #     new_population[i] = local_search(new_population[i])
    return new_population


population = create_population()
for generation in range(MAX_GENERATIONS):
    population = evolve_population(population)
    best_individual = max(population, key=lambda individual: objective_function(individual))
    print(f"Generation {generation}: Best Solution = {best_individual}, Best Fitness = {1/objective_function(best_individual)}")

