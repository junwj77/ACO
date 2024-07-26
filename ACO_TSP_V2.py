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
    def __init__(self, coords, distance_matrix, num_ants, alpha, beta, evaporation_rate, Q, iterations):
        self.coords = coords
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.iterations = iterations
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        self.visibility = 1 / (self.distance_matrix + 1e-10)
        self.best_route = None
        self.best_length = float('inf')

    def solve(self):
        for iteration in range(self.iterations):
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)
            self.plot_routes(all_routes, iteration)
            if iteration == self.iterations - 1:
                print(f"\nFinal Best Route Length = {self.best_length}")
                self.plot_best_route()

    def construct_solutions(self):
        all_routes = []
        for ant in range(self.num_ants):
            start = np.random.randint(self.num_cities)
            route = [start]
            current = start
            while len(route) < self.num_cities:
                if current not in route[:-1]:
                    move_probs = self.calc_move_probs(current, route)
                    next_city = np.random.choice(range(self.num_cities), p=move_probs)
                    route.append(next_city)
                    current = next_city
            route.append(start)
            route_length = self.calc_route_length(route)
            all_routes.append((route, route_length))
            if route_length < self.best_length:
                self.best_length = route_length
                self.best_route = route
        return all_routes

    def calc_move_probs(self, current, visited):
        probs = self.pheromone[current] ** self.alpha * (self.visibility[current] ** self.beta)
        mask = np.ones(len(probs), dtype=bool)
        mask[visited] = False
        probs = np.where(mask, probs, 0)
        return probs / np.sum(probs)

    def update_pheromones(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)
        for route, length in all_routes:
            for i in range(len(route) - 1):
                i, j = route[i], route[i + 1]
                self.pheromone[i, j] += self.Q / length

    def calc_route_length(self, route):
        return sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))

    def plot_routes(self, all_routes, iteration):
        if (iteration + 1) % 10 != 0:
            return
        plt.figure(figsize=(12, 10))
        for i in range(self.num_cities):
            for j in range(i,self.num_cities):
                if i != j:
                    plt.plot([self.coords[i][0], self.coords[j][0]], [self.coords[i][1], self.coords[j][1]], 'gray',
                             alpha=0.3)
                    mid_x = (self.coords[i][0] + self.coords[j][0]) / 2
                    mid_y = (self.coords[i][1] + self.coords[j][1]) / 2
                    plt.text(mid_x, mid_y, f"ph:{self.pheromone[i][j]:.2f},{self.pheromone[j][i]:.2f}\nd:{self.distance_matrix[i][j]:.2f}",
                             fontsize=8, ha='center', color='blue')

        for idx, coord in enumerate(self.coords):
            plt.text(coord[0], coord[1], str(idx), color="blue", fontsize=12)

        best_route_str = "Best Route so far: " + "->".join(
            map(str, self.best_route)) + f" with Total Distance: {self.best_length:.2f}"
        print("Final Best Route:", '->'.join(map(str, self.best_route)), f"\nwith Total Distance: {self.best_length:.2f}")
        route_str = " -> ".join(map(str, self.best_route)) + f" (Total Distance: {self.best_length:.2f})"

        plt.figtext(0.5, 0.01, best_route_str, ha="center", fontsize=12,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        plt.title(f'Routes at Iteration {iteration + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    def plot_best_route(self):
        if self.best_route is None:
            print("No best route to display.")
            return

        plt.figure(figsize=(12, 10))
        print("Final Best Route:", '->'.join(map(str, self.best_route)), f"with Total Distance: {self.best_length:.2f}")
        route_str = " -> ".join(map(str, self.best_route)) + f" (Total Distance: {self.best_length:.2f})"

        for i in range(len(self.best_route) - 1):
            start, end = self.best_route[i], self.best_route[i + 1]
            plt.plot([self.coords[start, 0], self.coords[end, 0]],
                     [self.coords[start, 1], self.coords[end, 1]], 'ro-')
            plt.text((self.coords[start, 0] + self.coords[end, 0]) / 2,
                     (self.coords[start, 1] + self.coords[end, 1]) / 2,
                     f"Pheromone: {self.pheromone[start][end]:.2f} d:{self.distance_matrix[start][end]:.2f}",
                     color='blue', fontsize=9)

        for idx, coord in enumerate(self.coords):
            plt.text(coord[0], coord[1], str(idx), color="red", fontsize=12)

        plt.title('Final Best Route Found')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.figtext(0.5, 0.01, route_str, ha="center", fontsize=12,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        plt.show()


# Example usage:
coords = load_tsp_data('data/burma_modified_7.tsp')
distance_matrix = calculate_distance_matrix(coords)
aco = ACO(coords=coords, distance_matrix=distance_matrix, num_ants=10, alpha=1, beta=1, evaporation_rate=0.5, Q=10, iterations=100)
aco.solve()
