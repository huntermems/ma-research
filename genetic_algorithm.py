import random
import numpy as np

# Genetic Algorithm Parameters
# n
NUMBER_OF_ROWS = 8
# m
NUMBER_OF_COLUMN = 20
# l
LENGTH_OF_STORAGE_BIN = 1
# h
HEIGHT_OF_STORAGE_BIN = 1
# vy
VERTICAL_VELOCITY = 2
# vx
HORIZONTAL_VELOCITY = 3
# vw
CROSS_VELOCITY = 1
# r
CURVE_LENGTH = 3
# w
RACK_DISTANCE = 4
# sc
INITIAL_SR_AISLE = 0


POPULATION_SIZE = 500
MAX_GENERATIONS = 1000
MUTATION_PROBABILITY = 0.5
CROSSOVER_PROBABILITY = 0.7

# Local Search Parameters
MAX_NO_IMPROVEMENT = 10

ITEM_NUMERATION = ['A', 'B', 'C', 'D', 'E']

current_aisle_of_sr = INITIAL_SR_AISLE

largest_aisle_to_be_visited = 0
smallest_aisle_to_be_visited = 0

aisle_rack_mapping = [(0,1), (2,3), (4,5), (6,7)]

warehouse = np.random.choice([*ITEM_NUMERATION, 0], size=(8, 4, 20), p=[0.05, 0.05, 0.05, 0.05, 0.05 ,0.75])
with open('warehouse.txt', 'w') as f:
    for row in warehouse:
        f.write(np.array2string(row, separator=', ', max_line_width=10000))
        f.write('\n\n')
index_of_items = list(zip(*np.where(warehouse != '0')))

item_location_mapping = {}

for item in ITEM_NUMERATION:
    item_locations = list(zip(*np.where(warehouse == item)))
    item_location_mapping[item] = item_locations

def t1(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 == current_aisle_of_sr:
        vertical_moving_time = (item[1] + 1) / VERTICAL_VELOCITY
        horizontal_moving_time = (item[2] + 1) / HORIZONTAL_VELOCITY
        total_time += 2 * round(max(vertical_moving_time, horizontal_moving_time),1)
    
    return total_time

def t2(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 != current_aisle_of_sr:
        aisle_travel_time = (NUMBER_OF_COLUMN * LENGTH_OF_STORAGE_BIN) / HORIZONTAL_VELOCITY
        curve_travel_time = 2 * CURVE_LENGTH / CROSS_VELOCITY
        total_time += aisle_travel_time + curve_travel_time
    return total_time

def t3(item):
    global current_aisle_of_sr
    total_time = 0
    if item[0] // 2 != current_aisle_of_sr:
        partial_horizontal_moving_time =  ((NUMBER_OF_ROWS - item[2] + 1) * LENGTH_OF_STORAGE_BIN) / HORIZONTAL_VELOCITY
        vertical_moving_time = (item[1] + 1) / VERTICAL_VELOCITY
        horizontal_moving_time = (item[2] + 1) / HORIZONTAL_VELOCITY
        total_time += partial_horizontal_moving_time + max(vertical_moving_time, horizontal_moving_time)
        current_aisle_of_sr = item[0] // 2
    return total_time

def t4():
    global largest_aisle_to_be_visited
    global smallest_aisle_to_be_visited
    distance = RACK_DISTANCE * (
        largest_aisle_to_be_visited - smallest_aisle_to_be_visited 
        + min(abs(current_aisle_of_sr - smallest_aisle_to_be_visited), abs(largest_aisle_to_be_visited - current_aisle_of_sr))
        )
    total_time = distance / CROSS_VELOCITY
    return total_time
    
# Objective Function
def total_t(solution):
    global current_aisle_of_sr
    global largest_aisle_to_be_visited
    global smallest_aisle_to_be_visited

    time = 0
    maximum_rack_number = max(solution, key= lambda x: x[0] )[0]
    minimum_rack_number = min(solution, key= lambda x: x[0] )[0]

    largest_aisle_to_be_visited = maximum_rack_number // 2
    smallest_aisle_to_be_visited = minimum_rack_number // 2
    cross_time = t4()
    for item in solution:
        time += t1(item) + t2(item) + t3(item)
    time += cross_time
    current_aisle_of_sr = INITIAL_SR_AISLE
    return time

def objective_function(solution):
    return 1/total_t(solution)

# Genetic Algorithm Functions
def create_individual():
    individual = []
    for item in ITEM_NUMERATION:
        item_locations = item_location_mapping[item]
        individual.append(item_locations[np.random.choice(len(item_locations))])
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
    if random.random() < MUTATION_PROBABILITY:
        # Randomly select the index of the gene
        indexes_of_gene_to_change = np.random.choice(len(individual))
        selected_gene = individual[indexes_of_gene_to_change]

        # Get the item name
        item_name = warehouse[selected_gene[0]][selected_gene[1]][selected_gene[2]]

        # Get the item's list of location
        item_locations = item_location_mapping[item_name]

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
    current_fitness = objective_function(*current_individual)
    no_improvement = 0
    while no_improvement < MAX_NO_IMPROVEMENT:
        neighbors = get_neighbors(current_individual)
        neighbor_fitnesses = [objective_function(*neighbor) for neighbor in neighbors]
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
        for offspring in [offspring1, offspring2]:
            if offspring is not None:
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

