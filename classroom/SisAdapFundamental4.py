import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        distances: Matriz de distancias entre ciudades (nodos)
        n_ants: Número de hormigas por iteración
        n_best: Cuántas de las mejores hormigas depositan feromona
        n_iterations: Número de iteraciones del algoritmo
        decay: Tasa de evaporación de feromona (0.0 a 1.0)
        alpha: Importancia de la feromona (rastro histórico)
        beta: Importancia de la heurística (distancia inversa)
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            
            # Encontrar el mejor camino de esta iteración
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            # Actualizar el mejor global si es necesario
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            
            # Evaporación de feromona (Adaptabilidad del sistema)
            self.pheromone * self.decay

        return all_time_shortest_path

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = [start]
        visited = set(path)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        path.append(start) # Volver al inicio para cerrar el ciclo
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0 # No visitar nodos ya visitados

        # Fórmula clave de ACO: (Feromona^alpha) * ((1/Distancia)^beta)
        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][path[i+1]]
        return total_dist

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for i in range(len(path) - 1):
                # Depositar feromona inversamente proporcional a la distancia
                self.pheromone[path[i]][path[i+1]] += 1.0 / self.distances[path[i]][path[i+1]]

# --- BLOQUE DE PRUEBA Y VISUALIZACIÓN ---

# 1. Generar ciudades aleatorias
num_ciudades = 15
coords = np.random.rand(num_ciudades, 2) * 100 # Coordenadas (x,y)

# 2. Crear matriz de distancias (Euclidiana)
dist_matrix = np.zeros((num_ciudades, num_ciudades))
for i in range(num_ciudades):
    for j in range(num_ciudades):
        dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        if i == j: dist_matrix[i][j] = np.inf # Evitar división por cero

# 3. Ejecutar Algoritmo
# alpha (peso feromona), beta (peso distancia), decay (evaporación)
ant_colony = AntColonyOptimizer(dist_matrix, n_ants=20, n_best=3, n_iterations=100, decay=0.95, alpha=1.2, beta=0.8)
best_path, best_score = ant_colony.run()

print(f"Distancia más corta encontrada: {best_score:.2f}")
print(f"Ruta: {best_path}")

# 4. Graficar
plt.figure(figsize=(10, 6))
plt.scatter(coords[:,0], coords[:,1], c='red', s=50, label='Ciudades')

# Dibujar la mejor ruta
x_path = [coords[i][0] for i in best_path]
y_path = [coords[i][1] for i in best_path]
plt.plot(x_path, y_path, c='blue', linestyle='-', linewidth=2, alpha=0.7, label='Ruta Hormiga')

plt.title(f"Optimización por Colonia de Hormigas (Dist: {best_score:.2f})")
plt.legend()
plt.grid(True)
plt.show()