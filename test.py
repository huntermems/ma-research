import numpy as np


class Particle:
    def __init__(self, cities):
        self.position = self.initialize_position(cities)
        self.velocity = np.random.permutation(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def initialize_position(self, cities):
        # Initialize a position with one post office per city
        position = np.random.permutation(len(cities))
        return position


def calculate_total_distance(position, distances):
    total_distance = 0
    num_cities = len(position)

    for i in range(num_cities - 1):
        city1 = position[i]
        city2 = position[i + 1]
        total_distance += distances[city1, city2]

    return total_distance


def update_velocity(particle, global_best_position, w=0.5, c1=1.5, c2=1.5):
    inertia = w * particle.velocity
    cognitive = c1 * np.random.rand(*particle.velocity.shape) * \
        (particle.best_position - particle.position)
    social = c2 * np.random.rand(*particle.velocity.shape) * \
        (global_best_position - particle.position)
    new_velocity = inertia + cognitive + social
    return new_velocity


def particle_swarm_optimization(num_particles, cities, distances, num_iterations):
    particles = [Particle(cities) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            total_distance = calculate_total_distance(
                particle.position, distances)
            if total_distance < particle.best_fitness:
                particle.best_fitness = total_distance
                particle.best_position = particle.position.copy()

            if total_distance < global_best_fitness:
                global_best_fitness = total_distance
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position)
            particle.position = np.argsort(particle.velocity)

    best_path = [cities[i] for i in global_best_position]

    return best_path, global_best_fitness


if __name__ == "__main__":
    # Define cities and distances (replace with actual distances between post offices)
    cities = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3',
              'B4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    # Example distance matrix (replace with actual distances between post offices)
    distances = np.random.rand(len(cities), len(cities))

    num_particles = 20
    num_iterations = 100

    best_path, best_fitness = particle_swarm_optimization(
        num_particles, cities, distances, num_iterations)

    print("Best Path:", best_path)
    print("Best Total Distance:", best_fitness)
