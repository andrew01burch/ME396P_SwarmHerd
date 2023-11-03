#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:39:12 2023

@author: tylercronin
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# dummy function to represent the environment reset
def reset_environment():
    pass #should be done before each iteration
    
# dummy function to represent taking an action in the environment
def step(action):
    pass 

# dummy function to calculate distance moved by an agent
def calculate_distance(movement):
    pass #can be used for all agents as well as the object

# dummy function to check if the task is completed
def is_task_completed():
    pass #should be a basic T/F

# Define the model
state_size = 10
action_size = 5
model = Sequential([
    Dense(128, activation='relu', input_shape=(state_size,)),
    Dense(64, activation='relu'),
    Dense(action_size, activation='linear')
])
model.compile(optimizer=Adam(lr=0.001), loss='mse')

# Define constants for the rewards and penalties
time_penalty = -1
movement_penalty = -0.01
obstacle_penalty = -10
completion_bonus = 1000

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    state = reset_environment()
    total_reward = 0
    total_movement = 0
    obstacles_touched = 0
    
    while True:
        
        agent1_action = #each agent performs an action
        agent2_action = ...
        agent3_action = ...
        
        next_state, done = step(action)
        movement = next_state_position - state_position
        distance_moved = calculate_distance(movement)
        total_movement += distance_moved
        
        # Check if an obstacle was touched
        if obstacle_or_wall_was_touched([True, False]):
            obstacles_touched += 1
            
        # Calculate reward
        reward = time_penalty
        reward += movement_penalty * distance_moved
        reward += obstacle_penalty * obstacles_touched
        if is_task_completed():
            reward += completion_bonus
            done = True
            
        # Update the model (this is a placeholder, replace with your actual update logic)
        current_state_target = reward  #calculated based on your algorithm
        model.fit(np.expand_dims(state, axis=0), np.array([current_state_target]), verbose=0)
        
        total_reward += reward
        state = next_state
        
        if done:
            break

    
    print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Total Movement: {total_movement}, Obstacles Touched: {obstacles_touched}')

print('Training complete!')



