import pygame
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32

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

# Constants for the simulation
max_velocity = 5.0  # The maximum velocity for particles.
max_acceleration = 2.0  # The maximum acceleration for particles.
n_particles = 20
state_size = n_particles * 4 + 4  # n_particles particles with (position, velocity) and (object position, target position).
action_size = n_particles * 2  # n_particles particles with a 2D force vector each.

# Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(state_size,)),
    Dense(64, activation='relu'),
    Dense(action_size, activation='tanh')  # Using tanh to keep the output in [-1, 1] range
])

optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer)

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

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer()

# making our particle object
class particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.force = force.astype(float)
        self.radius = float(radius)
        self.velocity = velocity.astype(float)
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

# Function to get the current state of the simulation as an input for the neural network
def get_state(particle_list, object, target_pos):
    # Convert all particle positions and velocities into a flat array
    particle_states = np.concatenate([p.position for p in particle_list] + [p.velocity for p in particle_list])
    object_state = object.position
    target_state = target_pos
    state = np.concatenate((particle_states, object_state, target_state))
    return state

# Function to apply the predicted actions to the particles
def apply_actions(actions, particle_list):
    # Scale the actions from [-1, 1] to the actual force range using max_acceleration
    scaled_actions = actions * max_acceleration
    
    # Apply the actions to the particles
    for i, particle in enumerate(particle_list):
        force = scaled_actions[i*2:(i+1)*2]
        # Update velocities with constraints
        particle.velocity += force / particle.mass
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
        # Update positions
        particle.physics_move()

# Function to calculate the reward based on the time taken and the distance to the target
def calculate_reward(start_time, current_time, object, target_pos):
    # Inverse reward based on time taken and distance to the target
    distance_to_target = np.linalg.norm(object.position - target_pos)
    time_taken = current_time - start_time
    reward = -distance_to_target / time_taken if time_taken > 0 else -distance_to_target
    return reward

# Function to perform a training step
def train_step(model, replay_buffer, batch_size, discount_factor = 1):
    if replay_buffer.size() < batch_size:
        return  # Not enough experiences to train on

    # Sample a minibatch of experiences from the replay buffer
    minibatch = replay_buffer.sample(batch_size)
    
    # Extract information from the data
    states = np.array([experience[0] for experience in minibatch])
    actions = np.array([experience[1] for experience in minibatch])
    rewards = np.array([experience[2] for experience in minibatch])
    next_states = np.array([experience[3] for experience in minibatch])
    dones = np.array([experience[4] for experience in minibatch])

    # Predict Q-values (targets) for starting state and next state
    target_qs = model.predict(states)
    next_target_qs = model.predict(next_states)
    
    # Update the targets with the observed rewards and the maximum Q-value for the next state
    for i in range(batch_size):
        # Find the index of the action taken. This assumes actions are discrete.
        action_index = np.argmax(actions[i])

        if dones[i]:
            target_qs[i][action_index] = rewards[i]
        else:
            target_qs[i][action_index] = rewards[i] + discount_factor * np.amax(next_target_qs[i])

    # Train the model on the states and the updated Q-values
    model.fit(states, target_qs, batch_size=batch_size, epochs=1, verbose=0)


# Function to reset the simulation
def reset_simulation(particle_list, object, object_initial_pos, n_particles, WIDTH, HEIGHT):
    pygame.init()
    running = True
    start_time = pygame.time.get_ticks()

    object.position = object_initial_pos.copy()
    object.velocity = np.zeros_like(object.velocity)
    for i in range(n_particles):
        particle_list[i].position = np.random.rand(2) * [WIDTH, HEIGHT]
        particle_list[i].velocity = np.random.rand(2)

# this is the thing we want to move tword the goal
object=particle(position=object_pos, radius=object_radius, mass = 50)

# creating a list of 20 particle objects all with random initial positions
particle_list=[]
for i in range (0,n_particles):
    instance=particle(position=np.random.rand(2) * [WIDTH, HEIGHT], mass = 10, velocity=np.random.rand(2))
    particle_list.append(instance)

# Before the simulation loop
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

# Main simulation loop
running = True
start_time = pygame.time.get_ticks()
while running:
    # screen.fill(WHITE)

    # # Draw border
    # pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)  # Border thickness of 2 pixels
    
    # unsure about this now... might be a part of reset?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Collisions handled here
    collective_force = np.zeros(2)
    for particle in particle_list:

        # friction_force = friction_coefficent * particle.velocity
        # particle.force += friction_force

        if np.linalg.norm(particle.position - object.position) <= contact_distance:

            particle.force+= (- object.position + particle.position)
            collective_force += (object.position - particle.position)
            
    object.force=collective_force + friction_coefficent * object.velocity

    handle_collisions(particle_list + [object])

    # Extract the current state
    current_state = get_state(particle_list, object, target_pos)

    #  Decide whether to take a random action or to predict one
    if np.random.rand() <= epsilon:
        # Take a random action: random force within the allowed range
        action = np.random.uniform(-max_acceleration, max_acceleration, action_size)
    else:
        # Predict action from the model
        action = model.predict(current_state.reshape(1, -1)).flatten()

    # Apply actions to the particles using physics_move
    apply_actions(action, particle_list)
    object.physics_move()

    # Update the state after applying actions
    next_state = get_state(particle_list, object, target_pos)

    # Calculate the current time and reward
    current_time = pygame.time.get_ticks()
    reward = calculate_reward(start_time, current_time, object, target_pos)
    done = np.linalg.norm(object.position - target_pos) < object_radius + target_radius

    # Store the experience in the replay buffer
    replay_buffer.add(current_state, action, reward, next_state, done)

    # Train the model
    train_step(model, replay_buffer, batch_size)

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Check if the episode is done (object reached the target)
    done = np.linalg.norm(object.position - target_pos) < object_radius + target_radius

    if done:
        # Reset the simulation to start a new episode
        pygame.quit()
        reset_simulation(particle_list, object, object_pos, n_particles, WIDTH, HEIGHT)
        start_time = pygame.time.get_ticks()  # Reset the timer
        # Optionally save the model
        # model.save('particle_swarm_model.h5')

    # try:
    #     pygame.draw.circle(screen, BLUE, center = (object.position[0], object.position[1]), radius=object.radius)
    # except:
    #     print("check")
    # pygame.draw.circle(screen, GREEN, target_pos, target_radius)
    # for particle in particle_list:
    #     pygame.draw.circle(screen, RED, center = (particle.position[0], particle.position[1]), radius=particle.radius)
    #  # Draw the cursor particle
    # pygame.draw.circle(screen, BLACK, cursor.position.astype(int), cursor.radius)
