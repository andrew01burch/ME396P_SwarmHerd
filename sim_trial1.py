import pygame
import numpy as np
import math


#change


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
pygame.display.set_caption('Simulation Evironment v1.0')
clock = pygame.time.Clock()

# Object and target settings
object_radius = 15
target_radius = 10
contact_distance = object_radius

object_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
target_pos = np.array([5* WIDTH // 6, HEIGHT // 2])

        # Normalize distance_vector to get collision direction
        collision_direction = (distance_vector / distance)

        # Move particles away based on their mass (heavier moves less)
        total_mass = float(self.mass + other_particle.mass)
        self.position += (overlap * (other_particle.mass / total_mass)) * collision_direction
        other_particle.position -= (overlap * (self.mass / total_mass)) * collision_direction    
        
    #maybe making a collide function here is a good idea, unsure as of now

#this is the thing we want to move tword the goal
object=particle(position=object_pos, raduis=object_radius, mass = 500)

#for now the cursor is treated like a particle so we can play around with physics, will remove later
cursor=particle(position=np.array(pygame.mouse.get_pos()), mass = 1)

#creating a list of 20 particle objects all with random initial positions
particle_list=[]
n_particles = 20
for i in range (0,n_particles):
    instence=particle(position=np.random.rand(2) * [WIDTH, HEIGHT], mass = 1)
    particle_list.append(instence)

def move_towards(point1, point2, speed=1.0):
    """Move point1 towards point2 by a certain speed."""
    direction = (point2 - point1)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:  # To prevent division by zero
        return point1
    direction = direction / direction_norm
    return point1 + direction * speed

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get cursor position and treat it as a particle
    cursor_pos = np.array(pygame.mouse.get_pos(), dtype=float)
    
    # Particles influenceing object position
    # Calculate the collective force from particles in contact
    collective_force = np.zeros(2)
    for particle in np.append(particle_positions, [cursor_pos], axis=0):  # Include cursor position
        if np.linalg.norm(particle - object_pos) <= contact_distance:
            # Particles push the object away from them, hence the negative sign
            collective_force = (object_pos - particle)

    # Apply the collective force to move the object
    if np.linalg.norm(collective_force) > 0:
        # The object's speed is proportional to the number of particles in contact
        object_pos += collective_force * (1 / np.linalg.norm(collective_force))


    # Update particle positions (just random movement) -> Eventually controlled by TF
    for i in range(n_particles):
        particle_positions[i] = move_towards(particle_positions[i], np.random.rand(2) * [WIDTH, HEIGHT], speed=1.0)

    # Draw everything
    try:
        pygame.draw.circle(screen, BLUE, center = (object.position[0], object.position[1]), radius=object.radius)
    except:
        print("check")
    pygame.draw.circle(screen, GREEN, target_pos, target_radius)
    for particle in particle_positions:
        pygame.draw.circle(screen, RED, particle.astype(int), particle_radius)
     # Draw the cursor particle
    pygame.draw.circle(screen, BLACK, cursor_pos.astype(int), particle_radius)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
