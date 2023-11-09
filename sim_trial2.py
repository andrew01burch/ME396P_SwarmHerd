import pygame
import numpy as np
import math

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

friction_coefficent = -0.05

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

def handle_collisions(particles, restitution_coefficient=1):
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            particle1, particle2 = particles[i], particles[j]
            distance_vector = particle1.position - particle2.position
            distance = np.linalg.norm(distance_vector).astype(float)
            if distance < (particle1.radius + particle2.radius):
                # Calculate overlap
                overlap = float((particle1.radius + particle2.radius) - distance)

                # Normalize distance_vector to get collision direction
                collision_direction = (distance_vector / distance)

                # Move particles away based on their mass (heavier moves less)
                total_mass = float(particle1.mass + particle2.mass)
                particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
                particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

                # Calculate relative velocity
                relative_velocity = particle1.velocity - particle2.velocity
                # Calculate velocity along the direction of collision
                velocity_along_collision = np.dot(relative_velocity, collision_direction)
                
                # Only proceed to update velocities if particles are moving towards each other
                if velocity_along_collision > 0:
                    # Apply the collision impulse
                    impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
                    particle1.velocity -= (impulse * particle2.mass) * collision_direction
                    particle2.velocity += (impulse * particle1.mass) * collision_direction


# making our particle object
class particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.force = force.astype(float)  # ensure force is also a float array
        self.radius = float(radius)
        self.velocity = velocity.astype(float)  # make sure velocity is float
        self.mass = float(mass)
    
    
    def collide(self, other_particle restitution_coefficient=1): #goal is to for the particles to do 
        distance_vector = self.position - other_particle.position
        distance = np.linalg.norm(distance_vector).astype(float)
        
         # Calculate overlap
        overlap = float((self.radius + other_particle.radius) - distance)

        # Normalize distance_vector to get collision direction
        collision_direction = (distance_vector / distance)

        # Move particles away based on their mass (heavier moves less)
        total_mass = float(self.mass + other_particle.mass)
        particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
        particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

        # Calculate relative velocity
        relative_velocity = particle1.velocity - particle2.velocity
        # Calculate velocity along the direction of collision
        velocity_along_collision = np.dot(relative_velocity, collision_direction)
        
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
        self.velocity += acceleration

        # Apply friction to the velocity
        self.velocity += friction_coefficent * self.velocity

        if np.linalg.norm(self.velocity) < 0.05:
            self.velocity = np.zeros_like(self.velocity)

        # Update position with velocity
        self.position += self.velocity
        
# this is the thing we want to move tword the goal
object=particle(position=object_pos, radius=object_radius, mass = 50)

# for now the cursor is treated like a particle so we can play around with physics, will remove later
cursor=particle(position=np.array(pygame.mouse.get_pos()), mass = 10)

# creating a list of 20 particle objects all with random initial positions
particle_list=[]
n_particles = 20
for i in range (0,n_particles):
    instance=particle(position=np.random.rand(2) * [WIDTH, HEIGHT], mass = 10, velocity=np.random.rand(2))
    particle_list.append(instance)

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Draw border
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)  # Border thickness of 2 pixels
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # updating the cursor particles position
    cursor.position=np.array(pygame.mouse.get_pos(), dtype=np.float64)
    
    # Collisions handled here
    collective_force = np.zeros(2)
    for particle in np.append(particle_list, cursor):  # Include cursor position in our particle list

        friction_force = friction_coefficent * particle.velocity
        particle.force += friction_force

        if np.linalg.norm(particle.position - object.position) <= contact_distance:

            particle.force+= (- object.position + particle.position)
            collective_force += (object.position - particle.position)
            
    object.force=collective_force + friction_coefficent * object.velocity

    handle_collisions(particle_list + [cursor])
    
    for particle in particle_list + [object]:
        particle.physics_move()

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