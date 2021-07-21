import numpy as np
import networkx as nx
import random
import timeit
import gym

env = gym.make('Taxi-v3')
n_actions = env.action_space.n
n_states = env.observation_space.n

qtable = np.zeros((n_states,n_actions))

n_episodes = 1000
n_steps = 100
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01
alpha = 0.7
gamma = 0.618
training_rewards_causal = []

G = nx.read_graphml('taxi_graph.graphml')

flag = 0

def get_node_values(observables):

	X = ['n0','n1','n2','n3','n4']
	Z = ['n5','n6']
	A = ['n7','n8']

	V = {key:0 for key in X+Z+A}
	
	taxi_x, taxi_y, passenger_pos, goal_pos = observables[0], observables[1], observables[2], observables[3]

	V['n0'] = passenger_pos
	V['n4'] = goal_pos
 
	V['n2'] = [taxi_x, taxi_y]
	if V['n2'] == [0,0]:
		V['n2'] = 0
	elif V['n2'] == [0,4]:
		V['n2'] = 1
	elif V['n2'] == [4,0]:
		V['n2'] = 2
	elif V['n2'] == [4,3]:
		V['n2'] = 3

	V['n1'] = int(V['n0'] == 4 or V['n0'] == V['n2'])
	V['n3'] = int(V['n6'] == V['n2'])
	V['n5'] = int(V['n0'] == 4)

	return V

def interventional_selection(s, G):

	observables = [i for i in env.decode(s)]
	#observables.reverse()
	V = get_node_values(observables)
	#print(V)
	Z = ['n5','n6']
	A = ['n7','n8']

	for z in Z:
		for n in G.predecessors(z):

			if(n in A):
				if(z == 'n5' and V['n5'] == 1):
					V[n] = 0					
					continue
				elif(z == 'n5' and V['n1'] == 1):
					V['n0'] = 4
					V['n5'] = 1
					V[n] = 1
					return 4

				if(V['n5'] == 1 and z == 'n6'):
					if(V['n3'] == 1):
						V[n] = 1
						V['n6'] = 1
						return 5
			
	return None

for episode in range(n_episodes):
	state = env.reset()
	total_reward = 0
	for step in range(n_steps):
		flag_a = 1
		action = interventional_selection(state, G)
		#print(action)	
		if(action == None):
			flag_a = 0
			if(epsilon > np.random.uniform(0,1)):
				while(action == None or action >= 4):
					action = env.action_space.sample()
				#print(action)
			else:
				action = np.argmax(qtable[state,:])
				#print(action)
		next_state, reward, done, info = env.step(action)
		total_reward += reward
		qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])
		state = next_state

		if(done):
			if(flag ==0 and total_reward >= 9):
				optimum_reward_ep = episode+1
				flag = 1
			#print('Episode_'+str(episode+1)+' Total reward: '+str(total_reward))
			#break
 	
	training_rewards_causal.append(total_reward)
	
	if(flag_a == 0):
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	
	#if(episode%10 == 0):
		#print('Average rolling reward: ' +str(sum(training_rewards[:-10])/10))

#print('Optimum reward received at epoch: ' +str(optimum_reward_ep))

import numpy as np
import random
import timeit
import gym
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
n_actions = env.action_space.n
n_states = env.observation_space.n

qtable = np.zeros((n_states,n_actions))

n_episodes = 1000
n_steps = 100
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01
alpha = 0.7
gamma = 0.618
training_rewards_epsilon = []

flag = 0

for episode in range(n_episodes):
	state = env.reset()
	total_reward = 0
	
	for step in range(n_steps):
		if(epsilon > np.random.uniform(0,1)):
			action = env.action_space.sample()
		else:
			action = np.argmax(qtable[state,:])

		next_state, reward, done, info = env.step(action)
		print(reward)
		total_reward += reward
		qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])
		state = next_state

		if(done):
			if(flag == 0 and total_reward >= 9):
				optimum_reward_ep = episode+1
				flag = 1

	training_rewards_epsilon.append(total_reward)
	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
	
	#if(episode%10 == 0):
		#print('Average rolling reward: ' +str(sum(training_rewards[:-10])/10))

#print('Optimum reward received at epoch: ' +str(optimum_reward_ep))

plt.plot([i for i in range(n_episodes)],training_rewards_causal, label = 'causal q-learning') 
plt.plot([i for i in range(n_episodes)], training_rewards_epsilon, label = 'q-learning')

plt.legend()

plt.xlabel('Episodes')
plt.ylabel('Reward')

plt.show()
