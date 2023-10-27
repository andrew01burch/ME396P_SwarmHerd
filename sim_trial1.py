import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Create the screen and clock objects
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Swarm Simulation')
clock = pygame.time.Clock()

# Object and target settings
object_radius = 15
target_radius = 10
object_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
target_pos = np.array([3 * WIDTH // 4, HEIGHT // 2])

# Particles settings
n_particles = 20
particle_radius = 5
particle_positions = np.random.rand(n_particles, 2) * [WIDTH, HEIGHT]

def move_towards(point1, point2, speed=1.0):
    """Move point1 towards point2 by a certain speed."""
    direction = (point2 - point1)
    direction = direction / np.linalg.norm(direction)
    return point1 + direction * speed

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Potential Field Movement
    object_pos = move_towards(object_pos, target_pos, speed=2.0)  # Attraction to target
    
    for particle in particle_positions:
        if np.linalg.norm(particle - object_pos) < 50:  # Within influence range
            object_pos = move_towards(object_pos, particle, speed=-2.0)  # Repulsion from particle

    # Update particle positions (just random movement for this example)
    for i in range(n_particles):
        particle_positions[i] = move_towards(particle_positions[i], np.random.rand(2) * [WIDTH, HEIGHT], speed=1.0)

    # Draw everything
    pygame.draw.circle(screen, BLUE, object_pos.astype(int), object_radius)
    pygame.draw.circle(screen, GREEN, target_pos, target_radius)
    for particle in particle_positions:
        pygame.draw.circle(screen, RED, particle.astype(int), particle_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
