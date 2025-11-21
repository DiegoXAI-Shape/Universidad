import matplotlib.pyplot as plt
import numpy as np

class AdaptiveTrafficLight:
    def __init__(self, min_green=10, max_green=60, learning_rate=0.5):
        self.queue = 0            # Coches esperando
        self.green_duration = 20  # Tiempo inicial del verde (s)
        self.min_green = min_green
        self.max_green = max_green
        self.learning_rate = learning_rate # Qué tan agresivo es el cambio (Alpha)
        
        # Historial para gráficas
        self.history_queue = []
        self.history_duration = []
        self.history_traffic_input = []

    def simulate_step(self, traffic_intensity):
        """Simula un ciclo completo de semáforo (Verde + Rojo)"""
        
        # 1. LLEGADA DE COCHES (Entorno)
        # Simulamos cuántos coches llegan durante este ciclo
        cars_arrived = np.random.poisson(traffic_intensity * (self.green_duration + 10)) 
        self.queue += cars_arrived
        self.history_traffic_input.append(cars_arrived)

        # 2. FLUJO DE SALIDA (Acción del Sistema)
        # Capacidad de la vía: asumiendo que pasa 1 coche cada 2 segundos de verde
        capacity = self.green_duration * 3.0 
        
        cars_passed = min(self.queue, capacity)
        self.queue -= cars_passed # Reducimos la fila
        
        # Guardamos datos
        self.history_queue.append(self.queue)
        self.history_duration.append(self.green_duration)

        # 3. ADAPTACIÓN (La parte "Complejilla")
        # Si quedaron coches (queue > 0), necesitamos más tiempo.
        # Si la fila se vació muy rápido (queue casi 0), sobra tiempo.
        
        # Error = Diferencia entre coches actuales y el objetivo (que es 0)
        error = self.queue - 0 
        
        # Ajuste proporcional (Controlador P)
        adjustment = error * self.learning_rate
        
        # Aplicamos el cambio respetando límites
        self.green_duration += adjustment
        self.green_duration = max(self.min_green, min(self.green_duration, self.max_green))

# --- SIMULACIÓN ---

sim = AdaptiveTrafficLight(learning_rate=0.8) # Learning rate alto para ver reacción rápida

print("Iniciando simulación de tráfico adaptativo...")

# Fase 1: Tráfico Normal (Ciclos 0-20)
for i in range(20):
    sim.simulate_step(traffic_intensity=0.5) # Intensidad media

# Fase 2: HORA PICO REPENTINA (Ciclos 21-40) - Aquí el sistema debe estresarse
print("¡ALERTA! Inicio de hora pico...")
for i in range(20):
    sim.simulate_step(traffic_intensity=2.1) # Intensidad brutal

# Fase 3: Calma (Ciclos 41-60) - El sistema debe relajarse
print("Fin de hora pico. Retornando a la calma...")
for i in range(20):
    sim.simulate_step(traffic_intensity=0.2)

# --- VISUALIZACIÓN ---

plt.figure(figsize=(12, 8))

# Subplot 1: Duración del Semáforo (La adaptación)
plt.subplot(2, 1, 1)
plt.plot(sim.history_duration, color='green', linewidth=2, label='Duración Luz Verde (s)')
plt.axvspan(20, 40, color='red', alpha=0.1, label='Hora Pico (Caos)')
plt.title('Adaptación del Sistema: Duración del Semáforo')
plt.ylabel('Segundos')
plt.legend()
plt.grid(True)

# Subplot 2: Cola de coches (El resultado)
plt.subplot(2, 1, 2)
plt.plot(sim.history_queue, color='blue', linestyle='--', label='Coches en Espera')
plt.plot(sim.history_traffic_input, color='orange', alpha=0.3, label='Entrada de Coches')
plt.axvspan(20, 40, color='red', alpha=0.1)
plt.title('Estado del Entorno: Congestión')
plt.xlabel('Ciclos de Tiempo')
plt.ylabel('Número de Coches')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()