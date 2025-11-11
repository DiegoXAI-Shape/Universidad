import pygame
import numpy as np
import sys

# Inicializa Pygame
pygame.init()

# Dimensiones de la pantalla
ancho, altura = 1000, 1000
pantalla = pygame.display.set_mode((ancho, altura))
pygame.display.set_caption("El Juego de la Vida de Conway")

# Definición de colores
COLOR_FONDO = (25, 25, 25)  # Casi negro
COLOR_REJILLA = (128, 128, 128)  # Gris para las líneas
COLOR_CELDA_VIVA = (255, 255, 255) # Amarillo brillante

# Define el tamaño de la rejilla (25x25)
n_celdas_x, n_celdas_y = 25, 25
dimCAnch = ancho / n_celdas_x
dimCAlt = altura / n_celdas_y

# Estado del juego: matriz de 0s y 1s. 0=Muerta, 1=Viva.
gameState = np.zeros((n_celdas_x, n_celdas_y))

# Control de la velocidad y estado
clock = pygame.time.Clock()
pausa_ejecucion = True # El juego inicia en pausa

while True:
    # Crea una copia del estado actual para calcular el siguiente estado
    newGameState = np.copy(gameState)

    # Limpia la pantalla
    pantalla.fill(COLOR_FONDO)
    
    # Control de FPS para que la simulación sea visible
    clock.tick(10) # 10 fotogramas por segundo (o "generaciones" por segundo)

    # --- Gestión de Eventos (OBLIGATORIO para que responda) ---
    for event in pygame.event.get():
        # Evento de cerrar la ventana
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # Evento de teclado (pausar/reanudar)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pausa_ejecucion = not pausa_ejecucion
        
        # Evento de clic de ratón (para revivir/matar celdas en pausa)
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Obtiene la posición del clic
            pos_x, pos_y = pygame.mouse.get_pos()
            
            # Convierte la posición de píxeles a coordenadas de celda
            celda_x, celda_y = int(np.floor(pos_x / dimCAnch)), int(np.floor(pos_y / dimCAlt))
            
            # Alterna el estado de la celda si el juego está en pausa
            if pausa_ejecucion:
                newGameState[celda_x, celda_y] = 1 if gameState[celda_x, celda_y] == 0 else 0
                # Actualiza inmediatamente para que el usuario vea el cambio
                gameState = np.copy(newGameState) 


    # --- Lógica del Juego de la Vida (Solo si NO está en pausa) ---
    if not pausa_ejecucion:
        
        # Iteramos sobre cada celda de la rejilla
        for y in range(0, n_celdas_y):
            for x in range(0, n_celdas_x):

                # Calculamos el número de vecinos VIVOS.
                # Se utiliza el módulo (%) para implementar bordes toroidales (la rejilla se envuelve).
                n_neigh = gameState[(x - 1) % n_celdas_x, (y - 1) % n_celdas_y] + \
                          gameState[(x)     % n_celdas_x, (y - 1) % n_celdas_y] + \
                          gameState[(x + 1) % n_celdas_x, (y - 1) % n_celdas_y] + \
                          gameState[(x - 1) % n_celdas_x, (y)     % n_celdas_y] + \
                          gameState[(x + 1) % n_celdas_x, (y)     % n_celdas_y] + \
                          gameState[(x - 1) % n_celdas_x, (y + 1) % n_celdas_y] + \
                          gameState[(x)     % n_celdas_x, (y + 1) % n_celdas_y] + \
                          gameState[(x + 1) % n_celdas_x, (y + 1) % n_celdas_y]
                
                # Regla 1: Muerte por soledad (menos de 2 vecinos)
                if gameState[x, y] == 1 and n_neigh < 2:
                    newGameState[x, y] = 0
                
                # Regla 2: Supervivencia (2 o 3 vecinos)
                elif gameState[x, y] == 1 and (n_neigh == 2 or n_neigh == 3):
                    newGameState[x, y] = 1 # Se mantiene viva
                
                # Regla 3: Muerte por sobrepoblación (más de 3 vecinos)
                elif gameState[x, y] == 1 and n_neigh > 3:
                    newGameState[x, y] = 0
                
                # Regla 4: Reproducción (una celda muerta con exactamente 3 vecinos vivos)
                elif gameState[x, y] == 0 and n_neigh == 3:
                    newGameState[x, y] = 1

    # --- Actualización y Dibujo de la Rejilla ---
    
    # Iteramos para dibujar el estado actual (sea el nuevo o el anterior si estaba en pausa)
    for y in range(0, n_celdas_y):
        for x in range(0, n_celdas_x):
            
            # Calculamos las coordenadas del polígono (cuadrado)
            poly = [((x) * dimCAnch, y * dimCAlt), 
                    ((x+1) * dimCAnch, y * dimCAlt),
                    ((x+1) * dimCAnch, (y + 1) * dimCAlt),
                    ((x) * dimCAnch, (y+1) * dimCAlt)]
            
            # Dibujamos la celda según su estado
            if newGameState[x, y] == 0:
                # Dibujamos la rejilla (celda muerta)
                pygame.draw.polygon(surface=pantalla, color=COLOR_REJILLA, points=poly, width=1)
            else:
                # Dibujamos una celda viva (sin borde, solo relleno)
                pygame.draw.polygon(surface=pantalla, color=COLOR_CELDA_VIVA, points=poly, width=0)

    # Actualiza el estado del juego para la próxima iteración
    gameState = np.copy(newGameState)
    
    # --- Interfaz de usuario (Mostrar estado) ---
    # Si está pausado, muestra un mensaje
    if pausa_ejecucion:
        font = pygame.font.Font(None, 74)
        texto = font.render("PAUSA", True, (255, 0, 0)) # Rojo
        rect_texto = texto.get_rect(center=(ancho // 2, altura // 2))
        # Fondo semi-transparente (Pygame no tiene soporte nativo, usamos un rectángulo)
        s = pygame.Surface((200, 100))
        s.set_alpha(128)  # Nivel de transparencia (0-255)
        s.fill((0, 0, 0)) # Color negro
        pantalla.blit(s, (ancho//2 - 100, altura//2 - 50))
        
        pantalla.blit(texto, rect_texto)

    # --- Actualización Final ---
    # Muestra todo lo dibujado en la pantalla
    pygame.display.flip()