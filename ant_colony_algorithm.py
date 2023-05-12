import numpy as np
import config

def ant_colony(num_ants, num_iterations, alpha, beta, rho, Q, distances):
    """
    Runs the Ant Colony Algorithm to find the shortest tour
    that visits all cities specified by the given distance matrix.
    
    Parameters:
        - num_ants: the number of ants to use
        - num_iterations: the number of iterations to run the algorithm for
        - alpha: the weight of the pheromone trail in ant decision making
        - beta: the weight of the distance between cities in ant decision making
        - rho: the pheromone decay rate
        - Q: the pheromone deposit amount
        - distances: a matrix of distances between cities
        
    Returns:
        - best_tour: the best tour found by the algorithm
        - best_distance: the distance of the best tour found by the algorithm
    """
    
    # Initialize pheromone matrix tau with all ones
    num_cities = distances.shape[0]
    tau = np.ones((num_cities, num_cities))

    print(tau)
    
    # Initialize best tour
    best_tour = None
    best_distance = float('inf')
    
    # Iterate over number of iterations
    for iteration in range(num_iterations):
        
        # Initialize ant tours and distances
        ant_tours = np.zeros((num_ants, num_cities), dtype=int)
        ant_distances = np.zeros(num_ants)
        
        # Iterate over ants
        for ant in range(num_ants):
            
            # Choose starting city at random
            current_city = np.random.randint(num_cities)
            visited_cities = set([current_city])
            ant_tours[ant, 0] = current_city
            
            # Construct tour by selecting next city based on pheromone and distance
            for i in range(1, num_cities):
                unvisited_cities = set(range(num_cities)) - visited_cities
                pheromone = tau[current_city, list(unvisited_cities)]
                distance = distances[current_city, list(unvisited_cities)]
                prob = np.power(pheromone, alpha) * np.power(1.0 / distance, beta)
                prob_norm = prob / np.sum(prob)
                next_city = np.random.choice(list(unvisited_cities), p=prob_norm)
                ant_tours[ant, i] = next_city
                ant_distances[ant] += distances[current_city, next_city]
                current_city = next_city
                visited_cities.add(current_city)
            
            # Update best tour
            if ant_distances[ant] < best_distance:
                best_tour = ant_tours[ant]
                best_distance = ant_distances[ant]
        
        # Update pheromone levels
        delta_tau = np.zeros((num_cities, num_cities))
        for ant in range(num_ants):
            for i in range(num_cities - 1):
                current_city, next_city = ant_tours[ant, i], ant_tours[ant, i + 1]
                delta_tau[current_city, next_city] += Q / ant_distances[ant]
            delta_tau[ant_tours[ant, -1], ant_tours[ant, 0]] += Q / ant_distances[ant]
        tau = (1 - rho) * tau + rho * delta_tau
    
    return best_tour, best_distance

# Define distance matrix
distances = np.array([[0, 2, 2, 5],
                      [2, 0, 4, 1],
                      [2, 4, 0, 3],
                      [5, 1, 3, 0]])

# Define parameters for the Ant Colony Algorithm
num_ants = 10
num_iterations = 100
alpha = 1.0
beta = 3.0
rho = 0.5
Q = 100.0

best_tour, best_distance = ant_colony(num_ants, num_iterations, alpha, beta, rho, Q, distances)

print(f"Best tour: {best_tour}")
print(f"Best distance: {best_distance}")