import pygame
import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from auxFunctionsAndObjects import handle_collisions
from auxFunctionsAndObjects import particle
import os

# Suppress TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')

#decide to visualize the game or not (for training purposes)
visualize = True

# Screen dimensions, need these if we are to visualize or not
WIDTH, HEIGHT = 800, 600
if visualize:
    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Screen dimensions

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
n_particles = 1 #lets try with 1 particle for now
friction_coefficient = -0.05
state_size = n_particles * 4 + 6  # position and velocity for each particle + object position and velocity + target position
action_size = n_particles * 2  # 2D force vector for each particle
learning_rate = 0.005
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 50  # Number of frames to wait before selecting a new action, made this 5 just to see how the model reacts
frame_counter = 0  # Counter to keep track of frames
collision_occurred = False
initial_force_magnitude = 10.0  # Adjust the magnitude of the initial force as needed
training_old_model = False

# Define the neural network for RL
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".keras"):
        model = tf.keras.models.load_model(f'{filename}')
        print(f'Using model: {filename}')
        training_old_model = True
    else:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(state_size,)),
            Dense(64, activation='relu'),
            Dense(action_size, activation='tanh')  # Force vector in range [-1, 1]
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate))

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
        force_magnitude = actions[i*2:(i+1)*2]
        particle.force = direction * force_magnitude

# Reward function emphasizing time and total movement


def reset_simulation(particle_list, object, sim_iter):
    #lets try having the object start in the same spot every time
    object.position = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    object.velocity = np.zeros_like(object.velocity)
    for particle in particle_list:
        #lets have our one particle start in a random spot
        particle.position = np.array([WIDTH // 5, HEIGHT // 5])
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
        model.fit(state.reshape(1, -1), target_f, epochs=1)  

# Initialize particle list and object
# Initialize particle list with initial force towards the object
particle_list = []
for _ in range(n_particles):
    # Random position for each particle
    position = np.array([WIDTH // 5, WIDTH // 5])

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

# Initialize previous_distance_to_target and particle distance to object
previous_distance_to_target = np.linalg.norm(object_pos - target_pos)
previous_particle_distance_to_object = np.linalg.norm(particle_list[0].position - object.position)


def calculate_reward(particle_list,
    object,
    target_pos,
    start_time, current_time, 
    collision_occurred_with_object,
    collision_occurred_between_particles,
    particle_distances_to_object,
    dela_distance_particle_object,
    previous_distance_to_target,
    delta_particle_distance_to_object):
    # Base components
    time_penalty = current_time - start_time
    #movement_penalty = sum(np.linalg.norm(p.velocity) for p in particle_list)
    #reward = -time_penalty 

    # Current distance between object and target
    distance_from_object_to_target = np.linalg.norm(object.position - target_pos)
    #change in disctnace between object and target
    delta_distance_to_target = previous_distance_to_target - distance_from_object_to_target
    #setting the new distance as the old distance to recalculate for the next loop
    previous_distance_to_target = distance_from_object_to_target
    #if delta_distance_to_target is negative, we are closer to the target and want to reward our model
    #reward = (delta_distance_to_target)*10
    
    
    reward = delta_particle_distance_to_object*10
    #print(delta_particle_distance_to_object)


    #print(reward)
    return reward #distance_from_object_to_target, having reward ONLY return reward, as far as I can tell 
                    #the other variable is used nowhere else in the code
# Define the maximum duration for a successful run (in milliseconds)
consecutive_successes = 0
max_success_frames = 600 #frames
reward = 0  # Initialize reward at zero

# Main simulation loop
running = True
start_time = pygame.time.get_ticks()
sim_iter = 1
particle_distances_to_object=[]
while running:
    frame_counter += 1
    if visualize:
        # Clear the screen and render the simulation
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Handle collisions and move particles
    handle_collisions(particle_list, object)

    if frame_counter % action_selection_frequency == 0:
        # the state you START at before taking this action
        current_state = get_state(particle_list, object, target_pos)
        if training_old_model == False:
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


    # Update physics of particles and object
    for particle in particle_list:
        particle.physics_move()
    object.physics_move()

    # keeping track of all of the distances between the object and the particles for each frame, I will
    #use this to calculate the change in distance between the particles and the object in order to reward/punish
    #my model accordingly.
    #CHANGED FROM ITERATING THROUGH ALL PARTICLES TO JUST THE FIRST PARTICLE IN PARTICLE_LIST BECASUE IN THIS
    #SCRIPT THERE IS ONLY 1 PARTICLE
    particle_distance_to_object = np.linalg.norm(particle_list[0].position - object.position)
    dela_distance_particle_object = previous_particle_distance_to_object - particle_distance_to_object
    # After updating the physics of particles and object


    # Calculate reward
    current_time = pygame.time.get_ticks()

    delta_particle_distance_to_object =  previous_particle_distance_to_object - particle_distance_to_object

    previous_particle_distance_to_object = particle_distance_to_object
    #we need to be RESETTING reward every time the agent chooses a new action, so reward should ADD UP for a
    #givin action, and then be reset when the agent takes a new action, then be ADDED up again for the next action
    reward= reward + calculate_reward(particle_list,
    object,
    target_pos,
    start_time, current_time, 
    collision_occurred_with_object,
    collision_occurred_between_particles,
    particle_distances_to_object,
    dela_distance_particle_object,
    previous_distance_to_target,
    delta_particle_distance_to_object)


    # Update previous_particle_distances for the next iteration
    #no longer needed becasue we are saving particledistances as a list
     #reward moving closer to the target:
    previous_particle_distance_to_object = particle_distance_to_object
    #set my current distance as my new previous distance
    
    if visualize:
        pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 2)
        pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
        pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

        for particle in particle_list:
            pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
        
        pygame.display.flip()
        clock.tick(60)

    

    #this checks to see if we have won
    done = np.linalg.norm(object.position - target_pos) < (object_radius + target_radius)
    # Store experience in the replay buffer
    if frame_counter % action_selection_frequency == 0:
        #next_state is the state that the action (calculated earlier) HAS TAKEN YOU TO.
        #we only need to calculate this when we are about to take a new action, becasue
        next_state = get_state(particle_list, object, target_pos)

        replay_buffer.add(current_state, action, reward, next_state, done)
        print(reward)
        reward = 0 #reset reward after adding to the replay buffer so we can calculate the reward for the next action

    # Train the model
    if done:
        consecutive_successes += 1
        if consecutive_successes >= 3:
            print("Model training completed.")
            model.save('particle_swarm_model.h5')
            running = False
        
        #Train model with accumulated experiences
        train_model(model, replay_buffer, batch_size, gamma)
        #Reset for new session
        print("Hey! That worked! Let's do it again!!")
        print(f'Reward: {reward}')
        frame_counter = 0
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle_list, object, sim_iter)
        
    elif frame_counter >= max_success_frames:
        print('That didnt quite work... lets try again.')
        #printing reward here is meaninlgess, as reward should be tied to a single action
        consecutive_successes = 0  # Reset if the task was not completed in time
        #Train model with accumulated experiences
        train_model(model, replay_buffer, batch_size, gamma)
        #Reset for new session
        frame_counter = 0
        frames, sim_iter, starting_distance_to_target = reset_simulation(particle_list, object, sim_iter)

    if sim_iter > 10:
        model.save('model_p10.keras')
        print("Model training completed.")
        running = False
if visualize:   
    pygame.quit()
