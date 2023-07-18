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

    def run(self, solution):
        best_solution = []
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
