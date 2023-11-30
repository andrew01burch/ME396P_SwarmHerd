# ME396P_SwarmHerd

This is the Final Project for Team G07 in ME396P Fall 2023.

Team members consist of Andrew Burch, Carl Stott, and Tyler Cronin.

This project aims to design and implement a simulated swarm of robot agents that employ neural networks to adaptively learn cooperative behaviors. By integrating swarm intelligence algorithms, the project aims to train the swarm to perform tasks such as object transport, exploration, and area coverage efficiently and autonomously.

Main problem: Implement adaptive learning mechanisms to accomplish cooperation among multiple entities without a centralized controller.

Additionally, we may expand into looking at constrained agents for additional uniqueness.

////FOLLOWING ARE THE NOTES FOR THE TA/INSTRUCTOR READING THIS FOR GRADING//////

Included are 2 files; the first is an example of the training simulation environment we have developed (sim_trial2.py). The mouse is treated as a particle and can be moved around the space to collide with other particles. This showcases the environment, which can be thought of behaving like an ice rink, and the particles like hockey pucks that elastically collide with each other (as in, the physics we have developed match this idea). The particles experience friction as they slide around in the space. We use proper Newtonian collision physics; the particles have a mass, and the coefficient of restitution of all these collisions is 1.

The second file (called Final_Full_Project.py) contains the full program that creates, simulates, and trains an RL model to try to solve the game of moving the blue object onto the green target by pushing a smaller particle into the object. There is a lot going on in this file, and it is heavily commented to help lead the viewer through what is happening. There are some sections of comments that explain crucial parts about the RL process and how it is implemented in the code, and those are sectioned off with /////read below///// and /////read above//////.

HOW TO USE:
1) Run Final_full_Project.py
2) When prompted, select yes/no to display the game (simulation runs much slower with the game shown)
3) When prompted, select explore/exploit to pick an exploratory stochastic policy (use for training) or an exploitative policy based on the predictions made by the model (used for validation). Picking explore will slowly over the cycle of the simulation begin to exploit the model's predictions as it trains.
4) After 10 training simulation iterations, the program will conclude. A model will automatically be saved and will automatically be reused and further trained the next time you run the Final_Full_project.py file.
5) Understanding what is printed out. Below is a snippet of what is printed out while the simulation is running:

////////////////////////////////////////////////////////////////////
SE
549.2946367511212
///////////////////////////////////////////////////////////////////
1/1 [==============================] - 0s 11ms/step
SW
123.11323949122622
/////////////////////////////////////////////////////////////////////

Each section separated by rows of /'s represents 1 decision cycle and the creation of 1 MDP (MDP's are explained in the code). In the first of the two decision cycles printed above, the first one shows that a stochastic policy was chosen (in the southeast direction) and that the total reward collected from taking that policy was 550. In the second cycle, the model chooses a policy (1/1 [==============================] - 0s 11ms/step denotes a model predicted policy). The model chose southwest, and the reward for picking that policy was 123. When the simulation is visualized, the printouts come out as the particle moves, allowing you to see how an action taken from a specific state resulted in a given reward.


                                    DISCUSSION ABOUT PROGRAM:
1) PROOF THAT OUR PROGRAM TRAINS THE MODEL TO PREDICT THE OPTIMAL POLICY FOR A GIVEN STATE (aka the program works):

Open the PROOF1.PNG file; you will see that the program has finished a simulation cycle and is in the middle of training. In this specific cycle of training, we want our model to learn from an MDP where the particle started at state (39,25) (the state space is explained in the final project script) and took the action 1 (move south), and received a -1134 reward for that action. Notice that the model's predicted potential value for picking action 1 BEFORE being refit with the knowledge that picking action 1 results in a reward of -1134 is 7.79. Now notice that AFTER being refit with this MDP, the model's predicted potential value for picking action 1 is 7.54. Meaning that the model learned that picking action 1 in state (39,25) results in a BAD reward, and the predicted potential value for picking that action in that state has been reduced. In PROOF2.PNG, I have run that same fitting multiple times, and now the predicted potential value for picking action 1 in state (39,25) is negative, showing that the model was further reinforced NOT to choose action 1 when in state (39,25).


2) But Carl, even after that explanation of why this code works, the models the script creates aren't taking actions that make sense, why is this?
    My answer to this question is that even with a single particle located in a discrete 2D state space and only operating from a discrete 8-dimensional action space, it takes a lot of training across a lot of data for an RL model to begin to predict good actions, and I did not have the processing power. Furthermore, I am unsure how to tell if the model is overfitting/underfitting, so the size and shape of the model I am using might also be wrong for this application.

3) Hey, what happened to using a swarm?
    The size of the state space and action space required to properly capture a single model that controls even 2 particles in moving the object is absolutely massive, meaning the amount of training data needed to fit this model would also be massive. Essentially, we didn't do a swarm because of the curse of dimensionality.