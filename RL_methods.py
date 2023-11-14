# Function to extract the current state
def get_state(particle, object, target_pos):
    particle_state = np.concatenate([particle.position, particle.velocity])
    object_state = np.concatenate([object.position, object.velocity])
    state = np.concatenate([particle_state, object_state, target_pos])
    return state

def apply_action(action, particle):
    particle.force = action  # Apply force to the particle

def calculate_individual_reward(particle, object, target_pos, collision_occurred_with_object, starting_distance_to_target):
    # Current distance between object and target
    distance_from_object_to_target = np.linalg.norm(object.position - target_pos)
    #change in disctnace between object and target
    delta_distance_to_target = starting_distance_to_target - distance_from_object_to_target
    #setting the new distance as the old distance to recalculate for the next loop
 
    #if delta_distance_to_target is negative, we are closer to the target and want to reward our model
    reward = (delta_distance_to_target)*100

    # Penalty for wall collisions
    if particle.hit_wall:
        reward -= 50  # Adjust the penalty value as needed

    # Reward for collision with object
    if collision_occurred_with_object:
        reward += 100  # Adjust the reward value as needed

    return reward

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

def train_model(model, replay_buffers, batch_size, gamma):
    # Train only if all buffers have enough samples
    if all([buffer.size() >= batch_size for buffer in replay_buffers]):
        # Sample from each buffer and train
        for buffer in replay_buffers:
            minibatch = buffer.sample(batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + gamma * np.amax(model.predict(next_state.reshape(1, -1), verbose = 0)[0]))
                target_f = model.predict(state.reshape(1, -1), verbose = 0)
                target_f[0][np.argmax(action)] = target
                model.fit(state.reshape(1, -1), target_f, epochs=1)  