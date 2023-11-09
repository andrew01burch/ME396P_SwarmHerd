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
friction_coefficent = -.05

object_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
target_pos = np.array([5* WIDTH // 6, HEIGHT // 2])

#making our particle object
class particle:
    def __init__(self, position, mass, raduis=5, velocity=np.array([0,0]), force=0, ):
        self.position=position
        self.force=force #the force acting on the particle
        self.radius=raduis
        self.velocity = velocity
        #velocity is 2 values [magnitude in x, magnitude in y]
        self.mass=mass
    
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
        
    def physics_move(self):
        # Collision with left or right boundary
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > WIDTH:
            self.velocity[0] = -self.velocity[0]
            self.position[0] = np.clip(self.position[0], self.radius, WIDTH - self.radius)
        # Collision with top or bottom boundary
        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > HEIGHT:
            self.velocity[1] = -self.velocity[1]
            self.position[1] = np.clip(self.position[1], self.radius, HEIGHT - self.radius)
            
        # Calculate acceleration from force
        acceleration = self.force / self.mass

        # Update velocity with acceleration
        self.velocity = self.velocity + acceleration

        # Apply friction to the velocity
        self.velocity = self.velocity +  friction_coefficent * self.velocity

        if np.linalg.norm(self.velocity) < 0.05:
            self.velocity = np.zeros_like(self.velocity)

        # Update position with velocity
        self.position = self.position + self.velocity
        
        
        
        
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
            collective_force = collective_force + (object.position - particle.position)
            
            #the object should push back on the particles, done below
            #particle.force = (particle.position - object.position) 
            
    #if multiple particles are colliding with the object, their forces should act
    #communitivly on the object, so collective force is summed across all active particles in the space
    # Particles push the object away from them, hence the negative sign.
    #object.force gives us a 2 length list of the x and y components of force acting on our object
    object.force=collective_force
    
    
    if np.linalg.norm(object.force) > 0:
        #use pythagorus to get the magnitide of that force
        force_magnitude=math.sqrt(collective_force[0]**2 + collective_force[1]**2)
        #get magnitude of velocity using intergral of f=ma
        velocity_magnitude= math.sqrt(2 * force_magnitude / object.mass)
        #now I need direction.. well the direction in the change of velocity will be the same as the direction of the force being applied to the systm..
        velocity_direction = object.force / np.linalg.norm(object.force)
        #now I need to update the velocity of my object. I need to scale velocity_direction up by velocity_magnitude, then add it to object.velocity
        delta_velocity = velocity_magnitude * velocity_direction
        #this SHOULD be my new object's velocity
        object.velocity = object.velocity + delta_velocity
    object.physics_move()


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
