import pygame
import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os

# Suppress TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')

# Screen dimensions
WIDTH, HEIGHT = 800, 600

visualize = True

if visualize:
    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

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
n_particles = 1
friction_coefficient = -0.05
state_size = n_particles * 4 + 4 + 2  # position and velocity for each particle + object position and velocity + target position
action_size = 1  # 2D force vector for each particle
learning_rate = 0.005
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 20  # Number of frames to wait before selecting a new action
frame_counter = 0  # Counter to keep track of frames
collision_occurred = False
initial_force_magnitude = 10.0  # Adjust the magnitude of the initial force as needed

# Define the neural network for RL
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".keras"):
        model = tf.keras.models.load_model(f'{filename}')
        print(f'Using model: {filename}')
    else:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(state_size,)),
            Dense(64, activation='relu'),
            Dense(4, activation='softmax')  # Output layer for four actions
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))
        

# Function to extract the current state
def get_state(particles, object, target_pos):
    # Include each particle's state and the object's state
    particle_states = np.concatenate([np.concatenate([p.position, p.velocity]) for p in particles])
    object_state = np.concatenate([object.position, object.velocity])
    state = np.concatenate([particle_states, object_state, target_pos])
    return state

# Function to apply actions to the particles
def apply_actions(action, particles, object):
    force_magnitude = 10.0  # You can adjust this value
    for particle in particles:
        if action == 0:  # Up
            particle.force = np.array([0, -force_magnitude])
        elif action == 1:  # Down
            particle.force = np.array([0, force_magnitude])
        elif action == 2:  # Left
            particle.force = np.array([-force_magnitude, 0])
        elif action == 3:  # Right
            particle.force = np.array([force_magnitude, 0])

# Reward function emphasizing time and total movement
def calculate_reward(particles, object, target_pos, collision_occurred_with_object, starting_distance_to_target):
    # Current distance between object and target
    distance_from_object_to_target = np.linalg.norm(object.position - target_pos)
    #change in disctnace between object and target
    delta_distance_to_target = starting_distance_to_target - distance_from_object_to_target
    #setting the new distance as the old distance to recalculate for the next loop
 
    #if delta_distance_to_target is negative, we are closer to the target and want to reward our model
    reward = (delta_distance_to_target)*10

    # Penalty for wall collisions
    for particle in particles:
        if particle.hit_wall:
            reward -= 25  # Adjust the penalty value as needed

    # Reward for collision with object
    if collision_occurred_with_object:
        reward += 2000  # Adjust the reward value as needed

    return reward

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

    if visualize:
        pygame.display.set_caption(f'Simulation Interation: {sim_iter}')
        # Clear the Pygame event queue to avoid processing stale events
        pygame.event.clear()

    steps = 0
    print(f'--- Simulation Interation #{sim_iter} ---')
    sim_iter+=1
    starting_distance_to_target = np.linalg.norm(object.position - target_pos)

    return steps, sim_iter, starting_distance_to_target

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
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose = 0)  

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
# collision_occurred_between_particles = False

#Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=50000)
batch_size = 32

# Initialize last chosen action
last_action = np.zeros(action_size)

# Define the maximum duration for a successful run (in milliseconds)
consecutive_successes = 0
max_success_frames = 600 #frames

# Initialize previous_distance_to_target
starting_distance_to_target = np.linalg.norm(object.position - target_pos)

# Main simulation loop
running = True
frames = 0
sim_iter = 1
# Main simulation loop
while running:
    frames += 1

    if visualize:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Handle collisions and move particles
    handle_collisions(particle_list, object)

    # Get current state
    current_state = get_state(particle_list, object, target_pos)

    # Decide actions for this frame
    if frames % action_selection_frequency == 0:
        try:
            if np.random.rand() <= epsilon:
                # Random action: choose from 0, 1, 2, 3 (up, down, left, right)
                action = np.random.choice([0, 1, 2, 3])
            else:
                # Model prediction
                action_probs = model.predict(current_state.reshape(1, -1)).flatten()
                action = np.argmax(action_probs)  # Choose the action with highest probability
            last_action = action
        except Exception as e:
            print(f"Error during model prediction: {e}")
    # Apply actions to all particles
    apply_actions(last_action, particle_list, object)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # Update physics of particles and object
    for particle in particle_list:
        particle.physics_move()
    object.physics_move()

    # Calculate next state and reward
    next_state = get_state(particle_list, object, target_pos)
    reward = calculate_reward(particle_list, object, target_pos, collision_occurred_with_object, starting_distance_to_target)

    # Visualization
    if visualize:
        pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)
        pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
        pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

        for particle in particle_list:
            pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
        
        pygame.display.flip()
        clock.tick(60)

    # Check for completion
    done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)

    # Store experience in the replay buffer
    replay_buffer.add(current_state, last_action, reward, next_state, done)

    # Train the model
    train_model(model, replay_buffer, batch_size, gamma)

    if done or frames >= max_success_frames:
        consecutive_successes = consecutive_successes + 1 if done else 0
        print(f'Iteration {sim_iter}: Reward - {reward}, Done - {done}, Frames - {frames}')
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle_list, object, sim_iter)

    if sim_iter > 10:
        model.save('model_p11.keras')
        print("Model training completed.")
        running = False

if visualize:
    pygame.quit()
