import pygame
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

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
pygame.display.set_caption('Simulation Evironment v2.0')
clock = pygame.time.Clock()

# Object and target settings
object_radius = 15
target_radius = 10
contact_distance = object_radius

# Initial positions
object_pos = np.array([WIDTH // 1, HEIGHT // 1], dtype=float)
target_pos = np.array([WIDTH // 2, HEIGHT //4])

# Hyperparameters
n_particles = 5
friction_coefficient = -0.01
state_size = n_particles * 4 + 4  # position and velocity for each particle + object position and velocity + target position
action_size = n_particles * 2  # 2D force vector for each particle
learning_rate = 0.001
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 2  # Number of frames to wait before selecting a new action
frame_counter = 0  # Counter to keep track of frames
collision_occurred = False
initial_force_magnitude = 0.1  # Adjust the magnitude of the initial force as needed


# Define the neural network for RL
model = Sequential([
    Dense(64, activation='relu', input_shape=(26,)),
    Dense(64, activation='relu'),
    Dense(action_size, activation='tanh')  # Force vector in range [-1, 1]
])
optimizer = Adam(learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# Function to extract the current state
def get_state(particle_list, object, target_pos):
    particle_states = np.concatenate([p.position for p in particle_list] + [p.velocity for p in particle_list])
    object_state = np.concatenate([object.position, object.velocity])
    state = np.concatenate([particle_states, object_state, target_pos])
    return state

# Function to apply actions to the particles
def apply_actions(actions, particle_list):
    for i, particle in enumerate(particle_list):
        force = actions[i*2:(i+1)*2]
        particle.force = force

# Reward function emphasizing time and total movement
def calculate_reward(particle_list, object, target_pos, start_time, current_time, collision_with_object, collision_between_particles):
    distance_to_target = np.linalg.norm(object.position - target_pos)
    time_penalty = current_time - start_time
    movement_penalty = sum(np.linalg.norm(p.velocity) for p in particle_list)
    
    # Adjust the reward components as necessary
    reward = -time_penalty - movement_penalty
    if collision_with_object:
        reward += 50  # Positive reward for collision with the object
    if collision_between_particles:
        reward -= 20  # Slight negative reward for collision between particles

    # Bonus for decreasing the distance to the target
    if distance_to_target < previous_distance_to_target:
        reward += 100  # Example bonus

    return reward, distance_to_target


# Class definition for particles
class particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.force = force.astype(float)
        self.radius = float(radius)
        self.velocity = velocity.astype(float)
        self.mass = float(mass)
        
    def physics_move(self):
        # Collision with boundaries and physics updates...
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
        self.velocity += friction_coefficient * self.velocity

        if np.linalg.norm(self.velocity) < 0.05:
            self.velocity = np.zeros_like(self.velocity)

        # Update position with velocity
        self.position += self.velocity

# Helper function to check if a collision occurs between two objects
def is_collision(particle1, particle2):
    distance = np.linalg.norm(particle1.position - particle2.position)
    return distance < (particle1.radius + particle2.radius)


def handle_collisions(particles, object, restitution_coefficient=1):
    global collision_occurred_with_object,  collision_occurred_between_particles
    collision_occurred_with_object = False
    collision_occurred_between_particles = False

    n = len(particles)

    # Check for collisions among particles
    for i in range(n):
        for j in range(i + 1, n):
            if is_collision(particles[i], particles[j]):
                collision_occurred_between_particles = True
            particle1, particle2 = particles[i], particles[j]
            if is_collision(particle1, particle2):
                # Normalize distance_vector to get collision direction
                distance_vector = particle1.position - particle2.position
                collision_direction = distance_vector / np.linalg.norm(distance_vector)
                total_mass = particle1.mass + particle2.mass

                # Calculate overlap
                overlap = (particle1.radius + particle2.radius) - np.linalg.norm(distance_vector)
                particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
                particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

                # Calculate relative velocity
                relative_velocity = particle1.velocity - particle2.velocity
                velocity_along_collision = np.dot(relative_velocity, collision_direction)

                # Only proceed to update velocities if particles are moving towards each other
                if velocity_along_collision > 0:
                    # Apply the collision impulse
                    impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
                    particle1.velocity -= (impulse * particle2.mass) * collision_direction
                    particle2.velocity += (impulse * particle1.mass) * collision_direction

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

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def train_model(model, replay_buffer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return  # Not enough samples for training

    minibatch = replay_buffer.sample(batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state.reshape(1, -1))[0]))
        target_f = model.predict(state.reshape(1, -1))
        target_f[0][np.argmax(action)] = target
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)



# Initialize particle list and object
# Initialize particle list with initial force towards the object
particle_list = []
for _ in range(n_particles):
    # Random position for each particle
    position = np.random.rand(2) * [WIDTH, HEIGHT]

    # Direction from particle to object
    direction_to_object = object_pos - position
    direction_to_object /= np.linalg.norm(direction_to_object)  # Normalize the direction

    # Set initial force towards the object
    initial_force = direction_to_object * initial_force_magnitude

    # Create particle with initial force
    new_particle = particle(mass=10, position=position, velocity=np.random.rand(2), force=initial_force)
    particle_list.append(new_particle)
object = particle(position=object_pos, radius=object_radius, mass=50)

collision_occurred_with_object = False
collision_occurred_between_particles = False

#Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=50000)
batch_size = 32

# Initialize last chosen action
last_action = np.zeros(action_size)


# Initialize previous_distance_to_target
previous_distance_to_target = float('inf')

# Main simulation loop
running = True
start_time = pygame.time.get_ticks()
while running:
    
    # Process input/events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # Handle collisions and move particles
    handle_collisions(particle_list , object)

    if frame_counter % action_selection_frequency == 0:
        # Get current state
        current_state = get_state(particle_list, object, target_pos)

        # Predict action from the model
        action = model.predict(current_state.reshape(1, -1)).flatten()
        
        last_action = action
    

        # Apply actions to particles
        apply_actions(last_action, particle_list)

    # Update frame counter
    frame_counter += 1

    # Update physics of particles and object
    for particle in particle_list:
        particle.physics_move()
    object.physics_move()
    
    # After updating the physics of particles and object
    next_state = get_state(particle_list, object, target_pos)
    done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)  # Example condition

    # Calculate the current distance to target
    current_distance_to_target = np.linalg.norm(object.position - target_pos)


    # Calculate reward
    current_time = pygame.time.get_ticks()
    reward, _ = calculate_reward(
    particle_list, 
    object, 
    target_pos, 
    start_time, 
    current_time, 
    collision_occurred_with_object, 
    collision_occurred_between_particles
    )
    
    #Store experience in an replay buffer
    replay_buffer.add(current_state, action, reward, next_state, done)

    # Train the model
    if collision_occurred_with_object or collision_occurred_between_particles:
        train_model(model, replay_buffer, batch_size, gamma)
        collision_occurred_with_object = False
        collision_occurred_between_particles = False


    # Clear the screen and render the simulation
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)  # Draw border
    pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
    pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

    for particle in particle_list:
        pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()