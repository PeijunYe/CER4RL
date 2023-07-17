# CER4RL
This project is for the paper Counterfactual Evolutionary Reasoning for Virtual Driver Reinforcement learning in Safe Driving

1、carla_env folder is the configuration of the train and test carla env.
2、PPO.py is the configuration of the Agent.
3、The main loop in main.py consists of four main functions:
	init(): This function is responsible for generating the initial sequence.
	rebuild_function(): This function is used for sequence reconstruction. Within this function, the mutate_policy_sampling function performs outlier sampling. 
	select(): This function implements the selection mechanism.
	collect_data(): This function is responsible for the learning process.
