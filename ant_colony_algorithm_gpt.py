import numpy as np
import random

class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, pheromone_weight, heuristic_weight, evaporation_rate):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone_weight = pheromone_weight
        self.heuristic_weight = heuristic_weight
        self.evaporation_rate = evaporation_rate
        self.distances = None
        self.pheromone_matrix = None

    def initialize(self, num_cities):
        self.distances = np.zeros((num_cities, num_cities))
        self.pheromone_matrix = np.ones((num_cities, num_cities))

    def add_distance(self, city1, city2, distance):
        self.distances[city1][city2] = distance
        self.distances[city2][city1] = distance

    def run(self):
        num_cities = len(self.distances)
        best_path = None
        best_path_length = np.inf

        for iteration in range(self.num_iterations):
            paths = []
            path_lengths = []

            for ant in range(self.num_ants):
                path = self.construct_path(num_cities)
                path_length = self.calculate_path_length(path)

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path.copy()

                paths.append(path)
                path_lengths.append(path_length)

            self.update_pheromone(paths, path_lengths)

        return best_path, best_path_length

    def construct_path(self, num_cities):
        start_city = random.randint(0, num_cities - 1)
        path = [start_city]
        visited = set([start_city])

        while len(visited) < num_cities:
            current_city = path[-1]
            next_city = self.select_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def select_next_city(self, current_city, visited):
        probabilities = self.calculate_probabilities(current_city, visited)
        return np.random.choice(len(probabilities), 1, p=probabilities)[0]

    def calculate_probabilities(self, current_city, visited):
        pheromone_sum = 0.0

        for city in range(len(self.distances)):
            if city not in visited:
                pheromone = self.pheromone_matrix[current_city][city] ** self.pheromone_weight
                heuristic = (1.0 / self.distances[current_city][city]) ** self.heuristic_weight
                pheromone_sum += pheromone * heuristic

        probabilities = []

        for city in range(len(self.distances)):
            if city in visited:
                probabilities.append(0.0)
            else:
                pheromone = self.pheromone_matrix[current_city][city] ** self.pheromone_weight
                heuristic = (1.0 / self.distances[current_city][city]) ** self.heuristic_weight
                probability = (pheromone * heuristic) / pheromone_sum
                probabilities.append(probability)

        return probabilities

    def calculate_path_length(self, path):
        length = 0.0

        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            length += self.distances[city1][city2]

        length += self.distances[path[-1]][path[0]]
        return length

    def update_pheromone(self, paths, path_lengths):
        self.pheromone_matrix *= (1.0 - self.evaporation_rate)

        for i, path in enumerate(paths):
            for j in range(len(path) - 1):
                city1 = path[j]
                city2 = path[j + 1]
                self.pheromone_matrix[city1][city2] += 1.0 / path_lengths[i]

# Example usage
aco = AntColonyOptimization(num_ants=10, num_iterations=100, pheromone_weight=1.0, heuristic_weight=2.0, evaporation_rate=0.1)

# Initialize the distances and pheromone matrix (example for 4 cities)
num_cities = 4
aco.initialize(num_cities)

# Add distances between cities (example distances)
aco.add_distance(0, 1, 10)
aco.add_distance(0, 2, 15)
aco.add_distance(0, 3, 20)
aco.add_distance(1, 2, 35)
aco.add_distance(1, 3, 25)
aco.add_distance(2, 3, 30)

# Run the algorithm
best_path, best_path_length = aco.run()

# Print the results
print("Best Path:", best_path)
print("Best Path Length:", best_path_length)
