import pygame
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_filename = 'particle_swarm_model.h5'
model = tf.keras.models.load_model(model_filename)

# Initialize pygame and set up the environment
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Model Visualization')
clock = pygame.time.Clock()

# Function to update the environment based on the model's actions
def update_environment():
    # Get the current state of the environment
    current_state = get_state(particle_list, object, target_pos)

    # Predict the action using the loaded model
    action = model.predict(current_state.reshape(1, -1), verbose = 0).flatten()

    # Apply the predicted action to the environment
    apply_actions(action, particle_list, object)

    # Update the physics of particles and the object
    for particle in particle_list:
        particle.physics_move()
    object.physics_move()

# Main loop for visualization
running = True
while running:
    screen.fill(WHITE)  # Clear the screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    update_environment()  # Update the environment based on the model's action

    # Render the environment
    for particle in particle_list:
        pygame.draw.circle(screen, RED, center=(particle.position[0], particle.position[1]), radius=particle.radius)
    pygame.draw.circle(screen, BLUE, center=(object.position[0], object.position[1]), radius=object.radius)
    pygame.draw.circle(screen, GREEN, target_pos.astype(int), target_radius)

    pygame.display.flip()  # Update the screen
    clock.tick(60)  # Control the frame rate

pygame.quit()
