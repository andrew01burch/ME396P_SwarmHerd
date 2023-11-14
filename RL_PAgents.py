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
n_particles = 10
friction_coefficient = -0.05
state_size = 4 + 4 + 2  # position and velocity for each particle + object position and velocity + target position
action_size = 2  # 2D force vector for each particle
learning_rate = 0.005
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 2  # Number of frames to wait before selecting a new action
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
            Dense(action_size, activation='tanh')  # Force vector in range [-1, 1]
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate))

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

#Initialize replay buffer
replay_buffers = [ReplayBuffer(capacity=50000) for _ in range(n_particles)]
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
while running:
    frames += 1
    if visualize:
    # Clear the screen and render the simulation
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Handle collisions and move particles
    handle_collisions(particle_list, object)

    # Gather all current states
    current_states = np.array([get_state(particle, object, target_pos) for particle in particle_list])

    # Batch prediction
    actions = model.predict(current_states, verbose=0)

    for particle_index, particle in enumerate(particle_list):
        action = actions[particle_index][:2]  # Assuming the model returns the correct shape
        apply_action(action, particle)
        particle.physics_move()  # Update physics of this particle

        # Calculate reward for this particle
        next_state = get_state(particle, object, target_pos)
        done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)
        reward = calculate_individual_reward(particle, object, target_pos, collision_occurred_with_object, starting_distance_to_target)

        # Add experience to the respective particle's replay buffer
        replay_buffers[particle_index].add(current_states[particle_index], actions[particle_index], reward, next_state, done)

    # Update object's physics
    object.physics_move()

    # Train the model with experiences from all buffers
    train_model(model, replay_buffers, batch_size, gamma)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    if visualize:
        pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)
        pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
        pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

        for particle in particle_list:
            pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
        
        pygame.display.flip()
        clock.tick(60)

    done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)

    # Define a training frequency
    training_freq = 50  # Example: Train the model every 50 frames

    # Inside the main loop
    if frames % training_freq == 0:
        train_model(model, replay_buffers, batch_size, gamma)

    # Train the model
    if done:
        consecutive_successes += 1
        if consecutive_successes >= 3:
            print("Model training completed.")
            model.save('particle_swarm_model.h5')
            running = False
        
        #Train model with accumulated experiences
        # train_model(model, replay_buffers, batch_size, gamma)
        #Reset for new session
        print("Hey! That worked! Let's do it again!!")
        print(f'Reward: {reward}')
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle_list, object, sim_iter)
        
    elif frames >= max_success_frames:
        print('That didnt quite work... lets try again.')
        print(f'Reward: {reward}')
        consecutive_successes = 0  # Reset if the task was not completed in time
        #Train model with accumulated experiences
        # train_model(model, replay_buffers, batch_size, gamma)
        #Reset for new session
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle_list, object, sim_iter)

    if sim_iter > 10:
        model.save('model_p10.keras')
        print("Model training completed.")
        running = False

if visualize:
    pygame.quit()
