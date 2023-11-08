#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:27:56 2023

@author: tylercronin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tf_agents.environments import utils
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
import pygame
import math
import gym
from gym import spaces

class SimulationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SimulationEnv, self).__init__()
        
        self.width, self.height = 800, 600 
        self.object_radius = 15
        self.target_radius = 10
        self.friction_coefficient = 0.5
        
        # Define observation space 
        #For our code, the observation space is all agent positions, 
        #their possoible velocities, the position of the object,
        #and the position of the objective
        self.observation_space = spaces.Dict({
            'particle_positions' : spaces.Box(low = 0, high = np.array([self.width, self.height]), shape = (20,2), dtype=np.float32),
            'particle_velocities' : spaces.Box(low = np.inf, high = np.inf, shape=(20, 2), dtype=np.float32),
            'object_position' : spaces.Box(low = 0, high=np.array([self.width, self.height]), shape=(1, 2), dtype=np.float32),
            'target_position' : spaces.Box(low = 0, high = np.array([self.width, self.height]), shape=(1, 2), dtype=np.float32)
        })
        
        #Define action space
        #For our code, this is the forces and movements that agents can apply
        #to the object, to simplify, this can be discrete (push up,down,left,right)
        self.action_space = spaces.Box(low = 10, high = 10, shape = (20, 2), dtype = np.float32)
        
        #Define environment parameters
        self.agent_mass = 1.0
        self.object_mass = 500.0
        self.target_pos = np.array([5 * self.width // 6, self.height // 2])
        self.object_pos = np.array([self.width // 2, self.height // 2], dtype = float)
        
        #Initialize State
        self.particle_positions = np.random.rand(20, 2) * [self.width, self.height]
        self.particle_velocities = np.zeros((20, 2), dtype=float) #Particles start with 0 velocity

        # Initialize reward structure
        self.time_penalty = -1  # Negative reward for each time step
        self.movement_penalty = -0.1  # Negative reward for each unit of movement

        # Done condition
        self.done_threshold = self.target_radius + self.object_radius  # Close enough to consider task completed


    def reset(self):
        # Reset the environment to its initial state
        # Reset the object's position to the center
        self.object_pos = np.array([self.width // 2, self.height // 2], dtype=float)

        # Reset particle positions and velocities
        self.particle_positions = np.random.rand(20, 2) * [self.width, self.height]
        self.particle_velocities = np.zeros((20, 2), dtype=float)  # Start with zero velocity

        # Reset the target position if it should be randomized, or keep it fixed
        self.target_pos = np.array([5 * self.width // 6, self.height // 2])

        # Combine all the observation components into a single dictionary
        observation = {
            'particle_positions': self.particle_positions,
            'particle_velocities': self.particle_velocities,
            'object_position': self.object_pos,
            'target_position': self.target_pos
        }

        return observation

    def step(self, action):
        # Apply the action and update the state of the environment
        #Also compute the reward and whether the episode is done
        for i, action in enumerate(actions):
            # Convert action to a numpy array if it's not
            action = np.array(action)

            # Update particle velocities based on the action
            self.particle_velocities[i] += action / self.agent_mass

            # Update particle positions
            self.particle_positions[i] += self.particle_velocities[i]

            # Apply friction
            friction_direction = -self.particle_velocities[i] / np.linalg.norm(self.particle_velocities[i])
            friction_force = self.friction_coefficient * self.agent_mass * friction_direction
            self.particle_velocities[i] += friction_force

        # Calculate the collective force from particles in contact with the object
        collective_force = np.zeros(2)
        for i, particle_pos in enumerate(self.particle_positions):
            if np.linalg.norm(particle_pos - self.object_pos) <= self.contact_distance:
                displacement = self.object_pos - particle_pos
                collective_force += displacement

        # Update object velocity and position based on collective force
        self.object_velocity += collective_force / self.object_mass
        self.object_pos += self.object_velocity

        # Apply friction to the object
        if np.linalg.norm(self.object_velocity) != 0:
            object_friction_direction = -self.object_velocity / np.linalg.norm(self.object_velocity)
            object_friction_force = self.friction_coefficient * self.object_mass * object_friction_direction
            self.object_velocity += object_friction_force

        # Update object position
        self.object_pos += self.object_velocity

        # Reward: negative for each time step and movement unit
        time_step_penalty = -1
        movement_penalty = -0.1 * np.sum(np.linalg.norm(actions, axis=1))
        reward = time_step_penalty + movement_penalty - np.linalg.norm(self.object_pos - self.target_pos)

        # Check for done condition
        done = np.linalg.norm(self.object_pos - self.target_pos) <= self.target_radius

        # Compile the new observation
        observation = {
            'particle_positions': self.particle_positions,
            'particle_velocities': self.particle_velocities,
            'object_position': self.object_pos,
            'target_position': self.target_pos
        }

        
        return observation, reward, done
    
    def render(self, mode='human', close = False):
        #Render the environment to the screen 
         # Check if Pygame is initialized, if not, initialize
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption('Simulation Environment')
            self.clock = pygame.time.Clock()
        
        # Fill the background
        self.screen.fill(WHITE)
        
        # Draw the target
        pygame.draw.circle(self.screen, GREEN, self.target_pos.astype(int), self.target_radius)
        
        # Draw the object
        pygame.draw.circle(self.screen, BLUE, self.object_pos.astype(int), self.object_radius)
        
        # Draw each particle
        for particle_pos in self.particle_positions:
            pygame.draw.circle(self.screen, RED, particle_pos.astype(int), self.particle_radius)
        
        # Update the full display
        pygame.display.flip()
        
        # Tick the clock
        self.clock.tick(60)  # Runs at 60 frames per second
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
            running = False

    
    def close(self):
        #Perform any necessary cleanup
        pygame.quit()


def create_actor_network(observation_space, action_space):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(observation_space,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space, activation='tanh')
    ])
    return model

def create_critic_network(observation_space, action_space, num_agents):
    total_obs_space = observation_space * num_agents
    total_action_space = action_space * num_agents

    input_obs = layers.Input(shape=(total_obs_space,))
    input_actions = layers.Input(shape=(total_action_space,))
    concat = layers.Concatenate()([input_obs, input_actions])

    out = layers.Dense(64, activation='relu')(concat)
    out = layers.Dense(64, activation='relu')(out)
    out = layers.Dense(1, activation='linear')(out)  # Q-value

    return models.Model([input_obs, input_actions], out)

n_agents = 20

actors = [create_actor_network(observation_space, action_space) for _ in range(n_agents)]
critics = [create_critic_network(observation_space, action_space, n_agents) for _ in range(n_agents)]
actor_optimizers = [tf.optimizers.Adam(1e-4) for _ in range(n_agents)]
critic_optimizers = [tf.optimizers.Adam(1e-3) for _ in range(n_agents)]
replay_buffer = []


# Neural network for the agent policy
def create_policy_network():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(observation_space,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space, activation='linear')
    ])
    return model

# Training process
def train_agent(env, agent):
    num_episodes = 1000  # Number of training episodes
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0 * n_agents
        done = False

        while not done:
            actions = [actors[i].predict(states[i][None, :]) for i in range(n_agents)]
            next_states, rewards, done, _ = env.step(actions)
            
            # Store experience in the replay buffer
            replay_buffer.append((states, actions, rewards, next_states, done))
            
            # Agent selects an action
            action = agent.model.predict(state)
            # Apply action to the environment
            next_state, reward, done, _ = env.step(action)
            # Agent learns from the action's outcome
            agent.learn(state, action, reward, next_state, done)
            # Update the state
            state = next_state
            episode_reward += reward

        print(f"Episode {episode} Reward: {episode_reward}")

# Define your custom environment
env = SimulationEnv()

# Check if the environment is compatible with TF-Agents
utils.validate_py_environment(env, episodes=5)

# Create the policy network
policy_network = create_policy_network()

# Create the TensorFlow agent
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=policy_network,
    optimizer=optimizers.Adam(learning_rate=1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss,
)

# Initialize the agent
agent.initialize()

# Train the agent
train_agent(env, agent)
