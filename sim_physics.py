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
        self.position += self.velocity

# Helper function to check if a collision occurs between two objects
def is_collision(particle1, particle2):
    distance = np.linalg.norm(particle1.position - particle2.position)
    return distance < (particle1.radius + particle2.radius)

def handle_collisions(particles, restitution_coefficient=1):
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            particle1, particle2 = particles[i], particles[j]
            distance_vector = particle1.position - particle2.position
            distance = np.linalg.norm(distance_vector).astype(float)
            if distance < (particle1.radius + particle2.radius):
                # Normalize distance_vector to get collision direction
                collision_direction = (distance_vector / distance)
                total_mass = float(particle1.mass + particle2.mass)
               
                overlap = float((particle1.radius + particle2.radius) - distance)
                particle1.position += (overlap * (particle2.mass / total_mass)) * collision_direction
                particle2.position -= (overlap * (particle1.mass / total_mass)) * collision_direction

                # Calculate relative velocity
                if distance_vector[0] > 0 or distance_vector[0] > 0:
                    relative_velocity = particle2.velocity - particle1.velocity
                else:
                    relative_velocity = particle1.velocity - particle2.velocity

                # Calculate velocity along the direction of collision
                velocity_along_collision = np.dot(relative_velocity, collision_direction)
                
                # Only proceed to update velocities if particles are moving towards each other
                if velocity_along_collision > 0:
                    # Apply the collision impulse
                    mass_factor = (2 * restitution_coefficient) / total_mass
                    impulse = velocity_along_collision * collision_direction * mass_factor
                    particle1.velocity += impulse * particle2.mass
                    particle2.velocity -= impulse * particle1.mass