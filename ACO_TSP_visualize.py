import numpy as np
import matplotlib.pyplot as plt


def load_tsp_data(filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')
    node_section = False
    coords = []
    for line in lines:
        if line.startswith('EOF'):
            node_section = False
        if node_section:
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))
        if line.startswith('NODE_COORD_SECTION'):
            node_section = True
    return np.array(coords)

def calculate_distance_matrix(coords):
    num_cities = len(coords)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

class ACO:
    def __init__(self, distance_matrix, num_ants, alpha, beta, evaporation_rate, Q, iterations, update_interval):
        self.coords = coords  # Store coordinates
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.iterations = iterations
        self.update_interval = update_interval
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # Initial pheromone levels
        self.visibility = 1 / (self.distance_matrix + 1e-10)
        self.best_route = None
        self.best_length = float('inf')


    def solve(self):
        length_history = []
        for iteration in range(self.iterations):
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)
            best_current_route, best_current_length = min(all_routes, key=lambda x: x[1])
            length_history.append(best_current_length)
            if best_current_length < self.best_length:
                self.best_length = best_current_length
                self.best_route = best_current_route
            if iteration == 0 or (iteration + 1) % self.update_interval == 0 or iteration == self.iterations - 1:
                self.plot_pheromones(iteration)

        self.plot_length_history(length_history)
        self.plot_best_route(self.coords)

        # Print the final best route length
        print("Final best route length:", self.best_length)

    def construct_solutions(self):
        all_routes = []
        for ant in range(self.num_ants):
            start = np.random.randint(self.num_cities)
            route = [start]
            current = start
            while len(route) < self.num_cities:
                move_probs = self.calc_move_probs(current, route)
                next_city = np.random.choice(range(self.num_cities), p=move_probs)
                route.append(next_city)
                current = next_city
            route.append(start)  # Returning to the start city to complete the tour
            all_routes.append((route, self.calc_route_length(route)))
        return all_routes

    def calc_move_probs(self, current, visited):
        probs = self.pheromone[current] ** self.alpha * self.visibility[current] ** self.beta
        probs[visited] = 0
        return probs / np.sum(probs)

    def update_pheromones(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)
        for route, length in all_routes:
            for i in range(len(route) - 1):
                i, j = route[i], route[i+1]
                self.pheromone[i][j] += self.Q / length

    def calc_route_length(self, route):
        return sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))

    def plot_pheromones(self, iteration):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.pheromone, interpolation='nearest', cmap='hot')
        plt.colorbar()
        plt.title(f'Pheromone Levels at Iteration {iteration + 1}')
        plt.xlabel('City j')
        plt.ylabel('City i')
        plt.show()

    def plot_length_history(self, length_history):
        plt.figure(figsize=(10, 6))
        plt.plot(length_history, marker='o')
        plt.title('Best Route Length Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Route Length')
        plt.show()

    def plot_best_route(self, coords):
        plt.figure(figsize=(8, 6))
        for i in range(len(self.best_route) - 1):
            start, end = self.best_route[i], self.best_route[i + 1]
            plt.plot([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]], 'ro-')
        plt.title('Best Route Found')
        plt.xlabel('City')
        plt.ylabel('City')
        plt.grid(True)
        plt.show()


#coords = load_tsp_data('data/berlin52.tsp')
#coords = load_tsp_data('data/ali535.tsp')
#coords = load_tsp_data('data/burma14.tsp')
coords = load_tsp_data('data/burma_modified_7.tsp')
#coords = load_tsp_data('d18512.tsp')
#coords = load_tsp_data('kroA100.tsp')
distance_matrix = calculate_distance_matrix(coords)

aco = ACO(distance_matrix=distance_matrix, num_ants=10, alpha=1, beta=1, evaporation_rate=0.5, Q=5, iterations=100, update_interval=10)
aco.solve()