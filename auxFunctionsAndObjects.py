import pygame
import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

WIDTH, HEIGHT = 800, 600
friction_coefficient = -0.05

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

# def handle_collisions(particles, object, restitution_coefficient=1):
#     global collision_occurred_with_object,  collision_occurred_between_particles
#     collision_occurred_with_object = False
#     collision_occurred_between_particles = False

#     n = len(particles)

#     # Check for collisions among particles
#     for i in range(n):
#         for j in range(i + 1, n):
#             if is_collision(particles[i], particles[j]):
#                 collision_occurred_between_particles = True
#             particle1, particle2 = particles[i], particles[j]
#             if is_collision(particle1, particle2):
#                 # Normalize distance_vector to get collision direction
#                 distance_vector = particle1.position - particle2.position
#                 collision_direction = distance_vector / np.linalg.norm(distance_vector)
#                 total_mass = particle1.mass + particle2.mass

#                 # Calculate overlap
#                 overlap = (particle1.radius + particle2.radius) - np.linalg.norm(distance_vector)
#                 particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
#                 particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

#                 # Calculate relative velocity
#                 relative_velocity = particle1.velocity - particle2.velocity
#                 velocity_along_collision = np.dot(relative_velocity, collision_direction)

#                 # Only proceed to update velocities if particles are moving towards each other
#                 if velocity_along_collision > 0:
#                     # Apply the collision impulse
#                     impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
#                     particle1.velocity -= (impulse * particle2.mass) * collision_direction
#                     particle2.velocity += (impulse * particle1.mass) * collision_direction

#     # Check for collisions between particles and the object
#     for particle in particles:
#         if is_collision(particle, object):
#             collision_occurred_with_object = True

#             # Collision handling between particle and object
#             distance_vector = particle.position - object.position
#             collision_direction = distance_vector / np.linalg.norm(distance_vector)
#             total_mass = particle.mass + object.mass

#             overlap = (particle.radius + object.radius) - np.linalg.norm(distance_vector)
#             particle.position += (overlap * (object.mass / total_mass)) * collision_direction
#             object.position -= (overlap * (particle.mass / total_mass)) * collision_direction

#             relative_velocity = particle.velocity - object.velocity
#             velocity_along_collision = np.dot(relative_velocity, collision_direction)

#             if velocity_along_collision > 0:
#                 impulse = (2 * velocity_along_collision / total_mass) * restitution_coefficient
#                 particle.velocity -= (impulse * object.mass) * collision_direction
#                 object.velocity += (impulse * particle.mass) * collision_direction
                
                
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
