# ME396P_SwarmHerd

This is the Final Project for Team G07 in ME396P Fall 2023.

Team members consist of Andrew Burch, Carl Stott, and Tyler Cronin.

This project aims to design and implement a simulated swarm of robot agents that employ neural networks to adaptively learn cooperative behaviors. By integrating swarm intelligence algorithms, the project aims to train the swarm to perform tasks such as object transport, exploration, and area coverage efficiently and autonomously.

Main problem: Implement adaptive learning mechanisms to accomplish cooperation out of multiple entities without a centralized controller.

Additionally, we may expand into looking at constrained agents for additional uniqueness.


////FOLLOWING ARE THE NOTES FOR THE TA/INSTRUCTOR READING THIS FOR GRADING//////

included are 2 files, the first is an example of the training simulation environment we have developed (sim_trial2.py). The mouse is treated as a particle and can be moved around the space to collide with other particles. This showcases the environment which can be thought of to behave like an ice rink and the particles
like hocky pucks that elastically collide with each other (as in, the physics we have developed match this idea).

the second file contains the full program that creates, simulates, and trains an RL model to try to solve the game of moving the blue object onto the green target by pushing a smaller particle into the object. 