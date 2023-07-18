import numpy as np
import random

class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, pheromone_weight, heuristic_weight, evaporation_rate):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.pheromone_weight = pheromone_weight
        self.heuristic_weight = heuristic_weight
        self.evaporation_rate = evaporation_rate
        self.pheromone_matrix = None

    def initialize(self, num_racks, num_rows, num_columns):
        num_vertices = num_racks * num_rows * num_columns
        self.pheromone_matrix = np.ones((num_vertices, num_vertices))

    def run(self, start_vertex, end_vertex):
        num_vertices = len(self.pheromone_matrix)
        best_path = None
        best_path_length = np.inf

        for iteration in range(self.num_iterations):
            paths = []
            path_lengths = []

            for ant in range(self.num_ants):
                path = [start_vertex]
                visited = set([start_vertex])
                path_length = 0.0

                while path[-1] != end_vertex:
                    current_vertex = path[-1]
                    probabilities = self.calculate_probabilities(current_vertex, visited, end_vertex)
                    next_vertex = self.select_next_vertex(probabilities)
                    path.append(next_vertex)
                    visited.add(next_vertex)
                    path_length += self.calculate_distance(current_vertex, next_vertex)

                paths.append(path)
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path

            self.update_pheromone(paths, path_lengths)

        return best_path, best_path_length

    def calculate_probabilities(self, current_vertex, visited, end_vertex):
        probabilities = []
        pheromone_sum = 0.0

        for vertex in range(len(self.pheromone_matrix)):
            if vertex not in visited:
                pheromone = self.pheromone_matrix[current_vertex][vertex] ** self.pheromone_weight
                heuristic = (1.0 / self.calculate_distance(current_vertex, vertex)) ** self.heuristic_weight
                probabilities.append(pheromone * heuristic)
                pheromone_sum += pheromone * heuristic
            else:
                probabilities.append(0.0)

        probabilities = [p / pheromone_sum for p in probabilities]
        return probabilities

    def select_next_vertex(self, probabilities):
        r = random.random()
        cumulative_prob = 0.0

        for i, probability in enumerate(probabilities):
            cumulative_prob += probability
            if cumulative_prob >= r:
                return i

    def update_pheromone(self, paths, path_lengths):
        self.pheromone_matrix *= (1.0 - self.evaporation_rate)

        for i, path in enumerate(paths):
            for j in range(len(path) - 1):
                vertex1 = path[j]
                vertex2 = path[j + 1]
                self.pheromone_matrix[vertex1][vertex2] += 1.0 / path_lengths[i]

    def calculate_distance(self, vertex1, vertex2):
        # You can define your own distance calculation between two vertices
        # based on the rack, row, and column information
        rack1, row1, col1 = self.decode_vertex(vertex1)
        rack2, row2, col2 = self.decode_vertex(vertex2)

        # Calculate the distance based on the rack, row, and column information
        distance = ...  # Your distance calculation logic here

        return distance

    def decode_vertex(self, vertex):
        num_rows, num_columns = self.pheromone_matrix.shape
        rack = vertex // (num_rows * num_columns)
        row = (vertex // num_columns) % num_rows
        col = vertex % num_columns
        return rack, row, col

    def encode_vertex(self, rack, row, col):
        num_rows, num_columns = self.pheromone_matrix.shape
        vertex = rack * (num_rows * num_columns) + row * num_columns + col
        return vertex


# Example usage
aco = AntColonyOptimization(num_ants=10, num_iterations=100, pheromone_weight=1.0, heuristic_weight=2.0, evaporation_rate=0.1)

# Initialize the pheromone matrix
num_racks = 5  # Example: 5 racks
num_rows = 10  # Example: 10 rows per rack
num_columns = 20  # Example: 20 columns per rack
aco.initialize(num_racks, num_rows, num_columns)

# Define the start and end vertices
start_rack = 0
start_row = 0
start_col = 0
start_vertex = aco.encode_vertex(start_rack, start_row, start_col)

end_rack = 2
end_row = 5
end_col = 10
end_vertex = aco.encode_vertex(end_rack, end_row, end_col)

# Run the algorithm
best_path, best_path_length = aco.run(start_vertex, end_vertex)

# Print the results
print("Best Path:", best_path)
print("Best Path Length:", best_path_length)