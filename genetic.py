import numpy as np
import random
import multiprocessing as mp

# Define cities and their coordinates
cities = {
    "A": (0, 0),
    "B": (1, 2),
    "C": (3, 4),
    "D": (5, 1),
    "E": (7, 3),
    "F": (6, 0),
    "G": (2, 1)
}

# Number of individuals in the population
POPULATION_SIZE = 50

# Number of generations
NUM_GENERATIONS = 100

# Crossover probability
CROSSOVER_PROB = 0.8

# Mutation probability
MUTATION_PROB = 0.2

# Number of cities
num_cities = len(cities)


# Function to calculate the distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate the total distance of a route
def total_distance(route):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(num_cities - 1)) + distance(cities[route[-1]], cities[route[0]])


# Function to create an initial population
def create_population():
    return [random.sample(cities.keys(), num_cities) for _ in range(POPULATION_SIZE)]


# Function to perform crossover (order-based crossover)
def crossover(parent1, parent2):
    child1, child2 = [-1] * num_cities, [-1] * num_cities

    # Select a random subset of parent genes to copy to children
    start, end = sorted(random.sample(range(num_cities), 2))
    child1[start:end + 1] = parent1[start:end + 1]
    child2[start:end + 1] = parent2[start:end + 1]

    # Fill the remaining genes using the other parent's order
    remaining1 = [gene for gene in parent2 if gene not in child1]
    remaining2 = [gene for gene in parent1 if gene not in child2]
    child1 = [gene if gene != -1 else remaining1.pop(0) for gene in child1]
    child2 = [gene if gene != -1 else remaining2.pop(0) for gene in child2]

    return child1, child2


# Function to perform mutation (swap two genes)
def mutate(route):
    i, j = random.sample(range(num_cities), 2)
    route[i], route[j] = route[j], route[i]


# Function to perform genetic algorithm on a population
def genetic_algorithm(population):
    for _ in range(NUM_GENERATIONS):
        # Calculate fitness (inverse of total distance)
        fitness = [1 / total_distance(route) for route in population]
        total_fitness = sum(fitness)

        # Create new population using selection and crossover
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.choices(population, weights=fitness, k=2)
            if random.random() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # Perform mutation on the new population
        for route in new_population:
            if random.random() < MUTATION_PROB:
                mutate(route)

        population = new_population

    # Find the best route (lowest total distance)
    best_route = min(population, key=total_distance)
    return best_route


# Function for parallel execution of the genetic algorithm on multiple populations
def parallel_genetic_algorithm(num_processes):
    pool = mp.Pool(processes=num_processes)
    populations = [create_population() for _ in range(num_processes)]
    results = pool.map(genetic_algorithm, populations)
    pool.close()
    pool.join()
    return min(results, key=total_distance)


if __name__ == "__main__":
    best_route = parallel_genetic_algorithm(num_processes=4)
    print("Best Route:", best_route)
    print("Total Distance:", total_distance(best_route))
