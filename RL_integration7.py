import pygame
import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Suppress TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Initialize pygame
pygame.init()
# Create the screen and clock objects
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Simulation Interation: 0')
clock = pygame.time.Clock()

# Object and target settings
object_radius = 15
target_radius = 10
contact_distance = object_radius

# Initial positions
object_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
target_pos = np.array([WIDTH // 2, HEIGHT //4])

# Exploration parameters
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration probability
epsilon_decay = 0.995  # Exponential decay rate for exploration prob

# Hyperparameters
n_particles = 20
friction_coefficient = -0.05
state_size = n_particles * 4 + 6  # position and velocity for each particle + object position and velocity + target position
action_size = n_particles * 2  # 2D force vector for each particle
learning_rate = 0.001
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 2  # Number of frames to wait before selecting a new action
frame_counter = 0  # Counter to keep track of frames
collision_occurred = False
initial_force_magnitude = 10.0  # Adjust the magnitude of the initial force as needed

# Define the neural network for RL
model = Sequential([
    Dense(64, activation='relu', input_shape=(state_size,)),
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
def apply_actions(actions, particle_list, object):
    for i, particle in enumerate(particle_list):
        # Calculate direction vector from particle to object
        direction = object.position - particle.position
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm  # Normalize the direction vector

        # Apply the new force to the particle
        # Assuming actions are now the force magnitudes
        force = actions[i*2:(i+1)*2]
        particle.force = force

# Reward function emphasizing time and total movement
def calculate_reward(particle_list, object, target_pos, start_time, current_time, collision_with_object, collision_between_particles, previous_particle_distances, current_particle_distances):
    # # Base components
    # time_penalty = current_time - start_time
    # #movement_penalty = sum(np.linalg.norm(p.velocity) for p in particle_list)
    # reward = -time_penalty 

    # Current distance to target
    distance_to_target = np.linalg.norm(object.position - target_pos)

    reward = 500 - distance_to_target * 1.5

    # Collision rewards
    if collision_with_object:
        reward += 200  # Reward for colliding with the object
    if collision_between_particles:
        reward -= 100  # Penalty for particle-particle collision

    # # Reward for moving towards the object
    # for prev_dist, curr_dist in zip(previous_particle_distances, current_particle_distances):
    #     distance_delta = prev_dist - curr_dist
    #     if distance_delta > 0:
    #         reward += 10 * distance_delta  # Scale reward based on improvement

    # Penalty for wall collisions
    wall_collision_penalty = sum(100 for p in particle_list if p.hit_wall)
    reward -= wall_collision_penalty

    return reward, distance_to_target

# Class definition for particles
class particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.force = force.astype(float)
        self.radius = float(radius)
        self.velocity = velocity.astype(float)
        self.mass = float(mass)
        self.hit_wall = False
        
    def physics_move(self):
        self.hit_wall = False
        # Collision with boundaries and physics updates...
        # Collision with left or right boundary
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > WIDTH:
            self.velocity[0] = -self.velocity[0]
            self.position[0] = np.clip(self.position[0], self.radius, WIDTH - self.radius)
            self.hit_wall = True
        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > HEIGHT:
            self.velocity[1] = -self.velocity[1]
            self.position[1] = np.clip(self.position[1], self.radius, HEIGHT - self.radius)
            self.hit_wall = True
            
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

def reset_simulation(particle_list, object, sim_iter):
    object.position = np.random.rand(2) * [WIDTH, HEIGHT]
    object.velocity = np.zeros_like(object.velocity)
    for particle in particle_list:
        particle.position = np.random.rand(2) * [WIDTH, HEIGHT]
        particle.velocity = np.zeros_like(particle.velocity)

    pygame.display.set_caption(f'Simulation Interation: {sim_iter}')
    sim_iter+=1

    start_time = pygame.time.get_ticks()

    # Clear the Pygame event queue to avoid processing stale events
    pygame.event.clear()

    return start_time, sim_iter


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
            target = (reward + gamma * np.amax(model.predict(next_state.reshape(1, -1), verbose = 0)[0]))
        target_f = model.predict(state.reshape(1, -1), verbose = 0)
        target_f[0][np.argmax(action)] = target
        model.fit(state.reshape(1, -1), target_f, epochs=5, verbose = 0)  

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
object = particle(position=object_pos, radius=object_radius, mass=5)

collision_occurred_with_object = False
collision_occurred_between_particles = False

#Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=50000)
batch_size = 32

# Initialize last chosen action
last_action = np.zeros(action_size)


# Define the maximum duration for a successful run (in milliseconds)
consecutive_successes = 0
max_success_duration = 15000  # 15 seconds in milliseconds


# Initialize previous_distance_to_target
previous_distance_to_target = np.linalg.norm(object_pos - target_pos)

# Initialize previous_particle_distances
previous_particle_distances = [np.linalg.norm(p.position - object.position) for p in particle_list]

# Main simulation loop
running = True
start_time = pygame.time.get_ticks()
sim_iter = 1
while running:

    # Clear the screen and render the simulation
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle collisions and move particles
    handle_collisions(particle_list, object)

    if frame_counter % action_selection_frequency == 0:
        # Get current state
        current_state = get_state(particle_list, object, target_pos)

        if np.random.rand() <= epsilon:
            # Choose a random force magnitude for exploration
            action = np.random.randn(n_particles * 2)  # Random values for each force dimension
        else:
            # Predict force magnitude based on model for exploitation
            action = model.predict(current_state.reshape(1, -1), verbose = 0).flatten()

        # Decay the epsilon value
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        last_action = action

    # Apply actions to particles
    apply_actions(last_action, particle_list, object)

    # Update frame counter
    frame_counter += 1

    # Update physics of particles and object
    for particle in particle_list:
        particle.physics_move()
    object.physics_move()

    # Check for wall collisions
    wall_collision = any(p.hit_wall for p in particle_list)

    # Calculate the current distances of particles to the object
    current_particle_distances = [np.linalg.norm(p.position - object.position) for p in particle_list]

    # After updating the physics of particles and object
    next_state = get_state(particle_list, object, target_pos)

    # Calculate the current distance to target
    current_distance_to_target = np.linalg.norm(object.position - target_pos)

    # Calculate reward
    current_time = pygame.time.get_ticks()
    reward, current_distance_to_target = calculate_reward(
        particle_list, 
        object, 
        target_pos, 
        start_time, 
        current_time, 
        collision_occurred_with_object, 
        collision_occurred_between_particles,
        previous_particle_distances,
        current_particle_distances
    )
    # Update previous_particle_distances for the next iteration
    previous_particle_distances = current_particle_distances
    
    
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)
    pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
    pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

    for particle in particle_list:
        pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
    
    pygame.display.flip()
    clock.tick(60)

    done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)
    # Store experience in the replay buffer
    replay_buffer.add(current_state, action, reward, next_state, done)

    # Train the model
    current_duration = pygame.time.get_ticks() - start_time
    if done:
        consecutive_successes += 1
        if consecutive_successes >= 3:
            print("Model training completed.")
            running = False
            model.save('particle_swarm_model.h5')
        
        #Train model with accumulated experiences
        train_model(model, replay_buffer, batch_size, gamma)
        #Reset for new session
        print("Hey! That worked! Let's do it again!!")
        print(f'Object distance to target: {current_distance_to_target}')
        start_time, sim_iter = reset_simulation(particle_list, object, sim_iter)
        
    elif current_duration >= max_success_duration:
        print('That didnt quite work... lets try again.')
        print(f'Object distance to target: {current_distance_to_target}')
        consecutive_successes = 0  # Reset if the task was not completed in time
        #Train model with accumulated experiences
        train_model(model, replay_buffer, batch_size, gamma)
        #Reset for new session
        start_time, sim_iter = reset_simulation(particle_list, object, sim_iter)
        
pygame.quit()
