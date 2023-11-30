import pygame
import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#from rl.agents import DDPGAgent #we use DDPG as our agent becasue we have a continuous action space and a continuous state space. see documentation: https://keras-rl.readthedocs.io/en/latest/agents/overview/
from rl.memory import SequentialMemory

from collections import deque
import random
import os

# Suppress TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')

#decide to visualize the game or not (for training purposes)
user_input = input("Do you want to visualize? (yes/no):")

# Set visualize to True if user input is 'yes', False otherwise.
visualize = user_input.lower() == 'yes'

#ask user if they want the model to train more or exploit more
user_input = input("Do you want your model to explore more or exploit more? (explore/exploit):")

#setting this to true will make the model explore more, setting it to false will make the model exploit more
training_old_model = user_input.lower() == 'exploit'

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
n_particles = 1 #the this exersise, we will have the agent control only 1 particle
friction_coefficient = -0.05
state_size = n_particles * 2  # state of a particle is its position, which is 2 values.
action_size = n_particles * 8  #each particle can take 8 possible actions, so the action size is 8
learning_rate = 0.005
gamma = 0.99  # Discount factor for future rewards
action_selection_frequency = 50  # Number of frames to wait before selecting a new action
frame_counter = 0  # Counter to keep track of frames
collision_occurred = False

# Define the neural network for RL. 
#the important things that we've learned about building networks is:
#1) make sure that the input layor is the same size as the state size
#2) make sure that the output layer is the same size as the action size
#3) MSE is the bess loss function for RL problems
def build_model(state_size, action_size):
    model = Sequential([
        Flatten(input_shape=(state_size,)),
        (Dense(24, activation='relu')),
        (Dense(24, activation='relu')),
        (Dense(action_size, activation='linear'))
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    return model


#if a network already exists in your directory, load it, otherwise create a new one
for filename in os.listdir(os.getcwd()):
    if filename.startswith("model_p"):
        model = tf.keras.models.load_model(f'{filename}')
        print(f'Using model: {filename}')
        
    else:
        model = build_model(state_size,action_size)




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
        self.position = self.position +  self.velocity

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

# Function to extract the current state
def get_state(particle_list, object, target_pos):
#/////////////////////READ BELOW/////////////////////////////
    #here I am disctizing the state space. I am doing that becasue I want my network to be able to recognize when
    #it is in a location that it has been in before, BUT with a continous state space that is near impossible
    #becasue the model will VERY UNLIKELY be in the EXACT same state twice. by discritizing 
    #the state space from a 800x600 pixel grid into 40x30 grid, the model can learn to recognize a wider range
    #of locations as being in the same "state". Why this matters: if the particle is at location
    #(150,260) in pixels, and it has learned from past experiance that taking the "up" action provides good reward 
    #at loation (151,262) in pixels, I want it to be able to recognize that going up at location (150,260) will also 
    #provide good reward and take the up action. if the state space is discritized, then (150,260) and (151,262)
    # WILL BE the same state, so the model will have seen that going up is the right choice at that state before,
    # and hopefully will make the same choice again.
#//////////////////////READ ABOVE////////////////////////////
    state = np.array([(particle_list[0].position[0] - 0) / (800 - 0) * 40, (particle_list[0].position[1] - 0) / (600 - 0) * 30])
    
    #this chops off the decimal places so we finish our discritization
    state = np.floor(state)
    return state

# Function to apply actions to the particles
def apply_actions(action, particle_list):
    force_magnitude = 1.0  # Adjust this value as needed
    diag_force_magnitude = np.sqrt(force_magnitude * 2)

    for particle in particle_list:
        if action == 0:  # north
            particle.force = np.array([0, -force_magnitude])
            print(f"N")

        elif action == 1:  # south
            particle.force = np.array([0, force_magnitude])
            print(f"S")

        elif action == 2:  # west
            particle.force = np.array([-force_magnitude, 0])
            print(f"W")

        elif action == 3:  # east
            particle.force = np.array([force_magnitude, 0])
            print(f"E")

        elif action == 4: #northwest
            particle.force = np.array([-diag_force_magnitude, -diag_force_magnitude])
            print(f"NW")
        
        elif action == 5: #northeast
            particle.force = np.array([diag_force_magnitude, -diag_force_magnitude])
            print(f"NE")
        
        elif action == 6: #southwest
            particle.force = np.array([-diag_force_magnitude, diag_force_magnitude])
            print(f"SW")
        
        elif action == 7: #southeast
            particle.force = np.array([diag_force_magnitude, diag_force_magnitude])
            print(f"SE")

        


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
    state_list.clear()

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
        target_f = model.predict(state.reshape(1, -1), verbose = 0)
        target_f[0][action] = target
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

    # Create particle with initial force
    new_particle = particle(mass=10, position=position)
    particle_list.append(new_particle)



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


    # Current distance between object and target
    distance_from_object_to_target = np.linalg.norm(object.position - target_pos)

    #change in disctnace between object and target
    delta_distance_to_target = previous_distance_to_target - distance_from_object_to_target

    #setting the new distance as the old distance to recalculate for the next loop
    previous_distance_to_target = distance_from_object_to_target

    #if delta_distance_to_target is negative, we are closer to the target and want to reward our model
    #reward = (delta_distance_to_target)*10
    
    #right now the ONLY thing that affects reward is how an action effects the distance between the particle and the obejct.
    #this is not ideal, but for now I want to show that I have a model that commands the particle to move tword the object.
    reward = delta_particle_distance_to_object*10


    #print(reward)
    return reward #distance_from_object_to_target, having reward ONLY return reward, as far as I can tell 
                    #the other variable is used nowhere else in the code



#the following statements surrounded by //'s are used to initalize variables before the game loop
#/////////////////////////////////////////////////////////////////////////////////////////
object = particle(position=object_pos, radius=object_radius, mass=50)

collision_occurred_with_object = False
collision_occurred_between_particles = False

#Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=50000)
batch_size = 10

# Initialize last chosen action
last_action = np.zeros(action_size)

# Initialize previous_distance_to_target and particle distance to object
previous_distance_to_target = np.linalg.norm(object_pos - target_pos)
previous_particle_distance_to_object = np.linalg.norm(particle_list[0].position - object.position)

# Define the maximum duration for a successful run (in milliseconds)
consecutive_successes = 0
max_success_frames = 6000 #this is the amount of frames that the simulation runs before we stop the simulation and train the model on the experiences it has had so far
reward = 0  # Initialize reward at zero
actionFrame = 0 #initalized so we dont crash on the first frame
state_list = []
running = True
start_time = pygame.time.get_ticks()
sim_iter = 1
particle_distances_to_object=[]
#/////////////////////////////////////////////////////////////////////////////////////////


# Main simulation loop
while running:
    
    if visualize:
        # Clear the screen and render the simulation
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Handle collisions and move particles
    handle_collisions(particle_list, object)

    # what this if statement does is makes sure we select a new action 
    #every action_selection_frequency frames, so in this case every 50 frames.
    #essentally what this means is that a force is applied to the particle every 50 frames
    #weather that force is stochastic or predicted by the model is determined by an if/else statement
    if frame_counter % action_selection_frequency == 0:
        actionFrame = frame_counter
        # the state you START at before taking this action
        state_list.append(get_state(particle_list, object, target_pos))

#/////////////////////READ BELOW/////////////////////////////
            #what this following if/else statement does is uses epsilon to decide if we are going to explore and pick a random
            #action, or if we are going to exploit and use our model's policy to predict an action. When the model is
            #first training, epsilon is 1, so we will always explore. As the model trains, epsilon decays, so
            #we will explore less and use the model's policy (AKA exploit) more. Ideally this means that our model will have a large amount
            #of experiences to learn from before we call on it to make any decisions, and then will start to 
            #exploit what it has learned when it has learned a policy from a large amount of explored experiences.
            #TLDR: we EXPLORE when we need to learn our system for data, and we EXPLOIT when we have learned a
            #policy that should be in the ballpark of preforming well.
            #essentally, the below statement is the decision-maker component of our agent.
#//////////////////////READ ABOVE////////////////////////////
        if training_old_model == False:
            if np.random.rand() <= epsilon:
                action = np.random.choice([0, 1, 2, 3])  # Random action
            else:
                #what this does is asks the model "givin the state I am in, what is the action with the
                # highest probability of being the best action?" and then the model returns a vector
                #of probabilities for each action, and we take the argmax of that vector to choose the 
                #action we take (if going up has the highest probability of giving the best reward, we go up)
                action_probs = model.predict(state_list[0].reshape(1, -1)).flatten()
                action = np.argmax(action_probs)
        else:
#/////////////////////READ BELOW/////////////////////////////
            #even if we start with a trained model, we STILL want SOME 
            #schochasticity to play a role in the choices we make, so that our model will explore 
            #and continue to learn. but becasue we are using a model that has already 
            #been trained in the past, we want to explore less and exploit more. 
            #so we will scale down epsilon when we are using a prevoiusly trained model.
            #dividing epsolon by 2 doubles the chance that we will exploit instead of explore.
            #technically this is a hyperparameter that we can tune, but at this point there
            #are so many hyperparamaters that I need to hold some of them constent.
#//////////////////////READ ABOVE////////////////////////////
            if np.random.rand() <= epsilon/2:
                action = np.random.choice([0, 1, 2, 3])  # Random action
            else:
                action_probs = model.predict(state_list[0].reshape(1, -1)).flatten()
                action = np.argmax(action_probs)

        # Decay the epsilon value
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        last_action = action

        # Apply actions to particles
        apply_actions(last_action, particle_list)

    #need to do this here or else we will be appending the same state to the state list twice
    frame_counter += 1

    # Update physics(position/velocity) of particles and object, this is the same as udapting the state of the system
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
    if frame_counter == 1:
        dela_distance_particle_object = 0
    # on the first frame we make sure that the model is not rewarded for the change in distance between the
    #particle and the object, as there is no previous distance to compare to


    # Calculate reward
    current_time = pygame.time.get_ticks()

    delta_particle_distance_to_object =  previous_particle_distance_to_object - particle_distance_to_object

    previous_particle_distance_to_object = particle_distance_to_object
    if frame_counter == 1:
        delta_particle_distance_to_object = 0
    

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

    # Store experience in the replay buffer, we want to do this the frame RIGHT BEFORE we take a new action.
    #this is the second part of our agent (essentally)
    if frame_counter == actionFrame + action_selection_frequency:
        #the below line gets us the state that the action we took action_selection_frequency ago TOOK us to
        state_list.append(get_state(particle_list, object, target_pos))
        if frame_counter ==1:
            action=np.zeros(action_size)
            #state_list[-2] is the state the particle was in BEFORE taking the action, and 
            #statelist[-1] is the state the particle is in AFTER taking the action
        replay_buffer.add(state_list[-2], action, reward, state_list[-1], done)
        state_list.clear()
        #only need to store the state before a taken action, and the state after a taken action
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
