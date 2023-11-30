# ME396P_SwarmHerd

This is the Final Project for Team G07 in ME396P Fall 2023.

Team members consist of Andrew Burch, Carl Stott, and Tyler Cronin.

This project aims to design and implement a simulated swarm of robot agents that employ neural networks to adaptively learn cooperative behaviors. By integrating swarm intelligence algorithms, the project aims to train the swarm to perform tasks such as object transport, exploration, and area coverage efficiently and autonomously.

Main problem: Implement adaptive learning mechanisms to accomplish cooperation out of multiple entities without a centralized controller.

Additionally, we may expand into looking at constrained agents for additional uniqueness.


////FOLLOWING ARE THE NOTES FOR THE TA/INSTRUCTOR READING THIS FOR GRADING//////

included are 2 files, the first is an example of the training simulation environment we have developed (sim_trial2.py). The mouse is treated as a particle and can be moved around the space to collide with other particles. This showcases the environment which can be thought of to behave like an ice rink and the particles
like hocky pucks that elastically collide with each other (as in, the physics we have developed match this idea).

the second file (called Final_Full_Project.py) contains the full program that creates, simulates, and trains an RL model to try to solve the game of moving the blue object onto the green target by pushing a smaller particle into the object. There is a lot going on in this file, and it is heavally commented to help lead the viewer through what is happening. There are some sections of comments that explain crucial parts about the RL process and how it is implimented in the code, and those are sectioned off with /////read below///// and 
/////read above//////.

HOW TO USE:
1) run Final_full_Project.py
2) when promped, select yes/no to display the game (simulation runs much slower with the game shown)
3) when primpted, select explore/exploit to pick an exploritory stchochastic policy (use for training) or an exploitative policy based on the predictions made by the model (used for validation). Picking explore will slowly over the sycle of the simulation begin to exploit the model's predictions as it trains.
4) after 10 training simulation iterations, the program will conclude. A model will automatically be saved and will automatically be reused and further trained the next time you run the Final-Full_project.py file.
5) understanding what is printed out. Below is a snippet of what is printed out while the simulation is running:


////////////////////////////////////////////////////////////////////
SE
549.2946367511212
///////////////////////////////////////////////////////////////////
1/1 [==============================] - 0s 11ms/step
SW
123.11323949122622
/////////////////////////////////////////////////////////////////////


each section seperated by rows of /'s represents 1 decision sycle and the creation of 1 MDP (MDP's are explained in the code). In the first of the two decision sycles printed above, the first one shows that a sthochastic policy was chosen (in the southeast direction) and that the total reward collected from taking that policy was 550. in the second cycle, the model chooses a policy (1/1 [==============================] - 0s 11ms/step denotes a model predicted policy). The model chose southwest, and the reward for picking that policy was 123. When the simulation is visualised, the printouts come out as the particle moves, allowing you to see how an action taken from a spesific state resulted in a givin reward.


6) PROOF THAT OUR PROGRAM TRAINS THE MODEL TO PREDICT THE OPTIMAL POLICY FOR A GIVIN STATE (aka the program works):

