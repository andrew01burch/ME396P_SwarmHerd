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
pygame.display.set_caption('Simulation Evironment v1.0')
clock = pygame.time.Clock()

# Object and target settings
object_radius = 15
target_radius = 10
contact_distance = object_radius

object_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
target_pos = np.array([5* WIDTH // 6, HEIGHT // 2])

#making our particle object
class particle:
    def __init__(self, position, particleForce=0, raduis=5):
        self.position=position
        self.particleForce=particleForce #the force acting on the particle, could be from the object or other particles
        self.radius=raduis

#making our object object... maybe this could just inherit from the particle class... for the future
class oject:
    def __init__(self, position=object_pos, objectForce=0, raduis = object_radius, velocity = 0):
        self.position=position
        self.objectForce=objectForce
        self.radius=raduis
        self.velocity = 0

#creating a list of 20 particle objects all with random positions
particle_list=[]
n_particles = 20
for i in range (0,n_particles):
    instence=particle(position=np.random.rand(1, 2) * [WIDTH, HEIGHT])
    particle_list.append(instence)

#function that moves the objects in out space
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
    cursor=particle(position=np.array(pygame.mouse.get_pos(), dtype=float))
    
    # Particles influenceing object position
    # Calculate the collective force from particles in contact
    collective_force = np.zeros(2)
    for particle in np.append(particle_list, cursor):  # Include cursor position in our particle list
        if np.linalg.norm(particle.position - object.position) <= contact_distance:
            #checking all particles that are in contact with the object
            # Particles push the object away from them, hence the negative sign
            collective_force = (object.position - particle.position)
            particle_force = (particle - object_pos)

    # Apply the collective force to move the object
    if np.linalg.norm(collective_force) > 0:
        # The object's speed is proportional to the number of particles in contact,
        #lets change it to be 1/4 as proportonal, as the perticles are 1/4 the size of the object
        object_pos += 1/4*(collective_force * (1 / np.linalg.norm(collective_force)))


    # Update particle positions (just random movement) -> Eventually controlled by TF
    for i in range(n_particles):
        particle_positions[i] = move_towards(particle_positions[i], np.random.rand(2) * [WIDTH, HEIGHT], speed=1.0)

    # Draw everything
    pygame.draw.circle(screen, BLUE, object_pos.astype(int), object_radius)
    pygame.draw.circle(screen, GREEN, target_pos, target_radius)
    for particle in particle_positions:
        pygame.draw.circle(screen, RED, particle.astype(int), particle_radius)
     # Draw the cursor particle
    pygame.draw.circle(screen, BLACK, cursor_pos.astype(int), particle_radius)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()