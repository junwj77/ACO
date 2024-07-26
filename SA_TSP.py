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


def plot_route(coords, route, iteration, best_length):
    plt.figure(figsize=(8, 6))
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        plt.plot([coords[start, 0], coords[end, 0]], [coords[start, 1], coords[end, 1]], 'ro-')
        plt.text((coords[start, 0] + coords[end, 0]) / 2,
                 (coords[start, 1] + coords[end, 1]) / 2,
                 f"{np.linalg.norm(coords[start] - coords[end]):.2f}", color='blue', fontsize=8)
    for idx, coord in enumerate(coords):
        plt.text(coord[0], coord[1], str(idx), color="blue", fontsize=12)
    plt.title(f'Best Route at Iteration {iteration} - Length: {best_length:.2f}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
    print(f"Best route at iteration {iteration}: {'-'.join(map(str, route + 1))} with length: {best_length:.2f}")

def plot_length_history(length_history):
    plt.figure(figsize=(10, 6))
    plt.plot(length_history, marker='o')
    plt.title('Route Length Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Route Length')
    plt.show()


def simulated_annealing(coords, iterations, temp, temp_decay, min_temp):
    num_cities = len(coords)
    current_route = np.random.permutation(num_cities)
    current_length = calc_route_length(current_route, coords)
    best_route = current_route.copy()
    best_length = current_length

    for i in range(iterations):
        if temp <= min_temp:
            break
        new_route = current_route.copy()
        start, end = sorted(np.random.choice(num_cities, 2, replace=False))
        new_route[start:end + 1] = new_route[start:end + 1][::-1]
        new_length = calc_route_length(new_route, coords)

        if new_length < best_length or np.random.rand() < np.exp((best_length - new_length) / temp):
            current_route = new_route
            current_length = new_length
            if new_length < best_length:
                best_length = new_length
                best_route = new_route.copy()

        temp *= temp_decay

        if i % 100 == 0:
            plot_route(coords, best_route, i, best_length)

    plot_route(coords, best_route, 'Final', best_length)
    print(f"Final best route: {'-'.join(map(str, best_route + 1))} with length: {best_length:.2f}")

def calc_route_length(route, coords):
    return sum(np.linalg.norm(coords[route[i]] - coords[route[(i + 1) % len(route)]]) for i in range(len(route)))

coords = load_tsp_data('data/burma_modified_7.tsp')

# Parameters and function call
simulated_annealing(coords, iterations=1000, temp=100, temp_decay=0.995, min_temp=0.1)


