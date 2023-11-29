import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os

tf.get_logger().setLevel('ERROR')

# Screen dimensions
WIDTH, HEIGHT = 800, 600
visualize = True

if visualize:
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Simulation Iteration: 0')
    clock = pygame.time.Clock()

object_radius = 15
target_radius = 10

object_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
target_pos = np.array([WIDTH // 2, HEIGHT // 4])

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

n_particles = 1
friction_coefficient = -0.05
state_size = n_particles * 4 + 4 + 2
action_size = 1
learning_rate = 0.005
gamma = 0.99
action_selection_frequency = 100
initial_force_magnitude = 10.0

#Neural Netowrk Setup
if os.path.exists('model.keras'):
    model = tf.keras.models.load_model('model.keras')
else:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

# Particle Class
class Particle:
    def __init__(self, mass=1.0, position=np.array([0.0, 0.0]), radius=5.0, velocity=np.array([0.0, 0.0]), force=np.array([0.0, 0.0])):
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.force = force.astype(float)
        self.mass = float(mass)
        self.radius = float(radius)
        self.hit_wall = False

    def update(self):
        acceleration = self.force / self.mass
        self.velocity += acceleration
        self.velocity *= (1 + friction_coefficient)
        self.position += self.velocity
        self.handle_boundaries()
        print(f"Particle position: {self.position}")

    def handle_boundaries(self):
        self.hit_wall = False
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > WIDTH:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], self.radius, WIDTH - self.radius)
            self.hit_wall = True
        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > HEIGHT:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], self.radius, HEIGHT - self.radius)
            self.hit_wall = True
        if self.hit_wall:
            print("Particle hit the wall.")

# Collision Handling
def is_collision(particle1, particle2):
    distance = np.linalg.norm(particle1.position - particle2.position)
    return distance < (particle1.radius + particle2.radius)

def handle_collisions(particle, object_push, restitution_coefficient=1):
    global collision_occurred_with_object
    collision_occurred_with_object = False

    if is_collision(particle, object_push):
        collision_occurred_with_object = True

        # Normalize distance vector to get collision direction
        distance_vector = particle.position - object_push.position
        collision_direction = distance_vector / np.linalg.norm(distance_vector)
        total_mass = particle.mass + object_push.mass

        # Calculate overlap
        overlap = (particle.radius + object_push.radius) - np.linalg.norm(distance_vector)
        particle.position += (overlap * (object_push.mass / total_mass)) * collision_direction
        object_push.position -= (overlap * (particle.mass / total_mass)) * collision_direction

        # Calculate relative velocity
        relative_velocity = particle.velocity - object_push.velocity
        velocity_along_collision = np.dot(relative_velocity, collision_direction)

        # Update velocities if particles are moving towards each other
        if velocity_along_collision > 0:
            impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
            particle.velocity -= (impulse * object_push.mass) * collision_direction
            object_push.velocity += (impulse * particle.mass) * collision_direction

# Reset the Simulation
def reset_simulation(particle, object_push, sim_iter):
    object_push.position = np.random.rand(2) * [WIDTH, HEIGHT]
    object_push.velocity = np.zeros_like(object_push.velocity)
    particle.position = np.random.rand(2) * [WIDTH, HEIGHT]
    particle.velocity = np.zeros_like(particle.velocity)

    if visualize:
        pygame.display.set_caption(f'Simulation Iteration: {sim_iter}')
        pygame.event.clear()

    starting_distance_to_target = np.linalg.norm(object_push.position - target_pos)
    return 0, sim_iter + 1, starting_distance_to_target

# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
    

# Function to train the model
def train_model(model, replay_buffer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return  # Not enough samples for training

    minibatch = replay_buffer.sample(batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward if done else (reward + gamma * np.amax(model.predict(next_state.reshape(1, -1), verbose=0)[0]))
        target_f = model.predict(state.reshape(1, -1), verbose=0)
        target_f[0][np.argmax(action)] = target
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)


# Function to decide actions
def choose_action(model, current_state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice([0, 1, 2, 3])  # Random action
    else:
        action_probs = model.predict(current_state.reshape(1, -1)).flatten()
        return np.argmax(action_probs)  # Choose the action with highest probability

# Function to apply actions to the particle
def apply_actions(action, particle):
    force_magnitude = 10.0  # Adjust this value as needed
    if action == 0:  # Up
        particle.force = np.array([0, -force_magnitude])
        print(f"Applied force: {particle.force}")

    elif action == 1:  # Down
        particle.force = np.array([0, force_magnitude])
        print(f"Applied force: {particle.force}")

    elif action == 2:  # Left
        particle.force = np.array([-force_magnitude, 0])
        print(f"Applied force: {particle.force}")

    elif action == 3:  # Right
        particle.force = np.array([force_magnitude, 0])
        print(f"Applied force: {particle.force}")

    

# Function to calculate reward
def calculate_reward(particle, object_push, target_pos, collision_occurred_with_object, starting_distance):
    distance_to_target = np.linalg.norm(object_push.position - target_pos)
    reward = (starting_distance - distance_to_target) * 10  # Reward for getting closer

    if particle.hit_wall:
        reward -= 25  # Penalty for hitting a wall

    if collision_occurred_with_object:
        reward += 2000  # Reward for pushing the object

    return reward

# Function to draw the environment
def draw_environment(screen, particle, object_push, target_pos, target_radius):
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)
    pygame.draw.circle(screen, BLUE, center=(int(object_push.position[0]), int(object_push.position[1])), radius=object_radius)
    pygame.draw.circle(screen, RED, center=(int(particle.position[0]), int(particle.position[1])), radius=particle.radius)
    pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

# Function to check if the simulation is completed
def check_completion(object_push, target_pos, object_radius, target_radius):
    return np.linalg.norm(object_push.position - target_pos) < (object_radius + target_radius)

# Function to print iteration results
def print_iteration_results(sim_iter, reward, done, frames):
    print(f'Iteration {sim_iter}: Reward - {reward}, Done - {done}, Frames - {frames}')

# Function to extract the current state
def get_state(particle, object_push, target_pos):
    particle_state = np.concatenate([particle.position, particle.velocity])
    object_state = np.concatenate([object_push.position, object_push.velocity])
    return np.concatenate([particle_state, object_state, target_pos])

# Initialize the object to be pushed
object_push = Particle(mass=5, position=object_pos, radius=object_radius, velocity=np.zeros(2), force=np.zeros(2))

# Initialize particle with initial force towards the object
particle = Particle(mass=10, position=np.random.rand(2) * [WIDTH, HEIGHT], velocity=np.random.rand(2), force=np.zeros(2))

# Initialize the replay buffer
replay_buffer = ReplayBuffer(capacity=50000)
batch_size = 32

# Initialize other simulation parameters
max_iterations = 50
last_action = 0

# Main simulation loop
running = True
frames = 0
sim_iter = 1
max_success_frames = 600  # Define the maximum number of frames for a successful run
max_iterations = 100  # Maximum number of iterations for the simulation
starting_distance_to_target = np.linalg.norm(object_push.position - target_pos)
while running:
    frames += 1

    if visualize:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Collision handling
    handle_collisions(particle, object_push)

    # Get current state
    current_state = get_state(particle, object_push, target_pos)

    # Decision making for actions
    if frames % action_selection_frequency == 0:
        action = choose_action(model, current_state, epsilon)
        apply_actions(action, particle)
        last_action = action
    
    print(f"Chosen action: {last_action}")

    # Update the physics
    particle.update()
    object_push.update()

    # Calculate next state and reward
    next_state = get_state(particle, object_push, target_pos)
    reward = calculate_reward(particle, object_push, target_pos, collision_occurred_with_object, starting_distance_to_target)

    # Visualization update
    if visualize:
        draw_environment(screen, particle, object_push, target_pos, target_radius)
        pygame.display.flip()
        clock.tick(60)

    # Check for completion and update the replay buffer
    done = check_completion(object_push, target_pos, object_radius, target_radius)
    replay_buffer.add(current_state, last_action, reward, next_state, done)

    # Train the model
    train_model(model, replay_buffer, batch_size, gamma)

    # Reset if necessary
    if done or frames >= max_success_frames:
        print_iteration_results(sim_iter, reward, done, frames)
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle, object_push, sim_iter)

    if sim_iter > max_iterations:
        model.save('model_final.keras')
        break

if visualize:
    pygame.quit()