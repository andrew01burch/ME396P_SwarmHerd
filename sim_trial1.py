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
    def __init__(self, position, raduis=5, velocity=0, force=0):
        self.position=position
        self.force=force #the force acting on the particle
        self.radius=raduis
        self.velocity = velocity
    
    #making moveing a method of the particle object, takes in current position
    #point we want to move to, and speed we want to move tword it
    def move_towards(self, point2, speed):
        """Move point1 towards point2 by a certain speed."""
        direction = (point2 - self.position)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:  # To prevent division by zero
            return self.position
        direction = direction / direction_norm
        newPose=self.position + direction * speed #add this to a particles position to update its location
        self.position=newPose
        
    #maybe making a collide function here is a good idea, unsure as of now

#this is the thing we want to move tword the goal
object=particle(position=object_pos, raduis=object_radius)

#for now the cursor is treated like a particle so we can play around with physics, will remove later
cursor=particle(position=np.array(pygame.mouse.get_pos()))

#creating a list of 20 particle objects all with random initial positions
particle_list=[]
n_particles = 20
for i in range (0,n_particles):
    instence=particle(position=np.random.rand(2) * [WIDTH, HEIGHT])
    particle_list.append(instence)


# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # updating the cursor particles position
    cursor.position=np.array(pygame.mouse.get_pos())
    
    # Particles influenceing object position
    # Calculate the collective force from particles in contact
    
    collective_force = np.zeros(2)
    for particle in np.append(particle_list, cursor):  # Include cursor position in our particle list
        if np.linalg.norm(particle.position - object.position) <= contact_distance:
            #checking all particles that are in contact with the object
            # Particles push the object away from them, hence the negative sign
            collective_force = collective_force + (object.position - particle.position)
            #if multiple particles are colliding with the object, their forces should act
            #communitivly on the object
            particle.force = (particle.position - object.position)
    object.force=collective_force
    
    
    # Apply the collective force to move the object
    if np.linalg.norm(collective_force) > 0:
        #unit vector in direction of collective force
        force_direction=collective_force * (1 / np.linalg.norm(collective_force))
        #making force magnitude of the particles 1/4 as effective, the particles are 1/4 the size of ght object
        force_magnitude=(1/4) * collective_force
        object.move_towards(force_direction, force_magnitude)
        #object.position = object.position + 1/4*(collective_force * (1 / np.linalg.norm(collective_force)))


    # Update particle positions (just random movement) -> Eventually controlled by TF
    for particle in particle_list:
        particle.move_towards(np.random.rand(2) * [WIDTH, HEIGHT], speed=1.0)

    # Draw everything
    try:
        pygame.draw.circle(screen, BLUE, center = (object.position[0], object.position[1]), radius=object.radius)
    except:
        print("check")
    pygame.draw.circle(screen, GREEN, target_pos, target_radius)
    for particle in particle_list:
        pygame.draw.circle(screen, RED, center = (particle.position[0], particle.position[1]), radius=particle.radius)
     # Draw the cursor particle
    pygame.draw.circle(screen, BLACK, cursor.position.astype(int), cursor.radius)


    pygame.display.flip()
    clock.tick(60)

pygame.quit()