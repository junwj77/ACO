import numpy as np

class ACO:
    def __init__(self, num_cities, num_ants, alpha, beta, evaporation_rate, Q, iterations):
        self.num_cities = num_cities
        self.num_ants = num_ants
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta    # Influence of heuristic information (inverse of distance)
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # Constant used in pheromone update
        self.iterations = iterations
        self.distance_matrix = np.random.rand(num_cities, num_cities) * 100
        self.pheromone = np.ones((num_cities, num_cities))  # Initial pheromone levels
        self.visibility = 1 / (self.distance_matrix + 1e-10)  # Heuristic information

    def solve(self):
        for iteration in range(self.iterations):
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)

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
            all_routes.append((route, self.calc_route_length(route)))
        return all_routes

    def calc_move_probs(self, current, visited):
        probs = self.pheromone[current] ** self.alpha * self.visibility[current] ** self.beta
        probs[visited] = 0  # Avoid revisiting cities
        return probs / np.sum(probs)

    def update_pheromones(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporation
        for route, length in all_routes:
            for i in range(len(route) - 1):
                i, j = route[i], route[i+1]
                self.pheromone[i][j] += self.Q / length

    def calc_route_length(self, route):
        return sum(self.distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))

# Parameters: num_cities, num_ants, alpha, beta, evaporation_rate, Q, iterations
aco = ACO(num_cities=10, num_ants=5, alpha=1, beta=2, evaporation_rate=0.5, Q=100, iterations=100)
aco.solve()
