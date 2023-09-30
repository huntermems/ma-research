import random
import config
import numpy as np

HEURISTIC_WEIGHT = 2
PHEROMONE_WEIGHT = 1


class AntColonyAlgorithm:

    number_vertices = len(config.ITEM_NUMERATION)
    pheromone_matrix = np.ones((number_vertices, number_vertices))

    def __init__(self, num_ants, num_iterations, pheromone_weight, heuristic_weight, evaporation_rate):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone_weight = pheromone_weight
        self.heuristic_weight = heuristic_weight
        self.evaporation_rate = evaporation_rate

    def calculate_probabilities(self, current_vertex, visited):
        probabilities = []
        pheromone_sum = 0.0

        for vertex in range(len(self.distances)):
            if vertex not in visited:
                pheromone = self.pheromone_matrix[current_vertex][vertex] ** self.pheromone_weight
                heuristic = (
                    1.0 / self.distances[current_vertex][vertex]) ** self.heuristic_weight
                probabilities.append(pheromone * heuristic)
                pheromone_sum += pheromone * heuristic
            else:
                probabilities.append(0.0)

        probabilities = [p / pheromone_sum for p in probabilities]
        return probabilities

    def select_next_vertex(self, current_vertex):
        pass

    def calculate_distance(current_vertex, next_vertex):
        pass

    def update_pheromone(paths, path_lengths):
        pass

    def run(self, solution):
        best_solution_path = None
        best_solution_path_length = np.inf
        for item in solution:
            best_path = None
            best_path_length = np.inf
            for _ in range(self.num_iterations):
                paths = []
                path_lengths = []

                for _ in range(self.num_ants):
                    path = [item]
                    visited = set([item])
                    path_length = 0.0
                while len(visited) < self.num_vertices:
                    current_vertex = path[-1]
                    probabilities = self.calculate_probabilities(
                        current_vertex, visited)
                    next_vertex = self.select_next_vertex(probabilities)
                    path.append(next_vertex)
                    visited.add(next_vertex)
                    path_length += self.calculate_distance(
                        current_vertex, next_vertex)

                paths.append(path)
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path

                self.update_pheromone(paths, path_lengths)
            if best_path and best_path_length < best_solution_path_length:
                best_solution_path_length = best_path_length
                best_solution_path = best_path
        return best_solution_path, best_solution_path_length
