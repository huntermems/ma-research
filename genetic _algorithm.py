import random
import math

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
MAX_GENERATIONS = 1000
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.7

# Local Search Parameters
MAX_NO_IMPROVEMENT = 10

# Objective Function
def objective_function(x, y):
    return -(x - 2) ** 2 + -(y + 3) ** 2

# Genetic Algorithm Functions
def create_individual():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    return (x, y)

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    return [objective_function(x, y) for x, y in population]

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    parents = random.choices(population, probabilities, k=2)
    return parents

def crossover(parents):
    x1, y1 = parents[0]
    x2, y2 = parents[1]
    if random.random() < CROSSOVER_PROBABILITY:
        x_new = (x1 + x2) / 2
        y_new = (y1 + y2) / 2
        return (x_new, y_new)
    else:
        return None

def mutate(individual):
    x, y = individual
    if random.random() < MUTATION_PROBABILITY:
        x_new = x + random.uniform(-0.5, 0.5)
        y_new = y + random.uniform(-0.5, 0.5)
        return (x_new, y_new)
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
        offspring = crossover(parents)
        if offspring is not None:
            mutation1 = mutate(offspring)
            mutation2 = mutate(offspring)
            if mutation1 is not None:
                new_population.append(mutation1)
            if mutation2 is not None:
                new_population.append(mutation2)
    new_population.extend(population[:POPULATION_SIZE - len(new_population)])
    for i in range(len(new_population)):
        new_population[i] = local_search(new_population[i])
    return new_population

population = create_population()
for generation in range(MAX_GENERATIONS):
    population = evolve_population(population)
    best_individual = min(population, key=lambda individual: objective_function(*individual))
    print(f"Generation {generation}: Best Solution = {best_individual}, Best Fitness = {objective_function(*best_individual)}")

