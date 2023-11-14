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
# target_pos = np.array([5* WIDTH // 6, HEIGHT // 2])
target_pos = np.array([WIDTH, HEIGHT])

# Helper function to check if a collision occurs between two objects
def is_collision(particle1, particle2):
    distance = np.linalg.norm(particle1.position - particle2.position)
    return distance < (particle1.radius + particle2.radius)

def handle_collisions(particles, object, restitution_coefficient=1):
    global collision_occurred_with_object, collision_occurred_between_particles
    collision_occurred_with_object = False
    collision_occurred_between_particles = False

    def update_positions_and_velocities(particle1, particle2, is_object=False):
        distance_vector = particle1.position - particle2.position
        collision_direction = distance_vector / np.linalg.norm(distance_vector)
        total_mass = particle1.mass + particle2.mass

        overlap = (particle1.radius + particle2.radius) - np.linalg.norm(distance_vector)
        particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
        particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

        relative_velocity = particle1.velocity - particle2.velocity
        velocity_along_collision = np.dot(relative_velocity, collision_direction)

        if velocity_along_collision > 0:
            impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
            particle1.velocity -= (impulse * particle2.mass) * collision_direction
            particle2.velocity += (impulse * particle1.mass) * collision_direction

    # Check for collisions among particles
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            if is_collision(particles[i], particles[j]):
                collision_occurred_between_particles = True
                update_positions_and_velocities(particles[i], particles[j])

    # Check for collisions between particles and the object
    for particle in particles:
        if is_collision(particle, object):
            collision_occurred_with_object = True
            update_positions_and_velocities(particle, object, is_object=True)

    # Check for collisions between particles and the object
    for particle in particles:
        if is_collision(particle, object):
            collision_occurred_with_object = True

            # Collision handling between particle and object
            distance_vector = particle.position - object.position
            collision_direction = distance_vector / np.linalg.norm(distance_vector)
            total_mass = particle.mass + object.mass

            overlap = (particle.radius + object.radius) - np.linalg.norm(distance_vector)
            particle.position += (overlap * (object.mass / total_mass)) * collision_direction
            object.position -= (overlap * (particle.mass / total_mass)) * collision_direction

            relative_velocity = particle.velocity - object.velocity
            velocity_along_collision = np.dot(relative_velocity, collision_direction)

            if velocity_along_collision > 0:
                impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
                particle.velocity -= (impulse * object.mass) * collision_direction
                object.velocity += (impulse * particle.mass) * collision_direction


# making our particle object
class particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.force = force.astype(float)  # ensure force is also a float array
        self.radius = float(radius)
        self.velocity = velocity.astype(float)  # make sure velocity is float
        self.mass = float(mass)
        
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
    cursor.velocity = [10, 10] 

    handle_collisions(particle_list, object)
    
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
    clock.tick(120)

pygame.quit()