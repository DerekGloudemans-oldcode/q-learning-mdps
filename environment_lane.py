# import required packages
import numpy as np
import random


########################## Environment_pred Class ##############################
# Environment classes provide functions for maintaining agents, the environment, 
# and the interaction between them
# Environment_pred represents the predator CMDP as discussed in Laetitia et al.
class Environment_lane():
    
    #_________________________Environment_lane()______________________________#
    # initializes a 1-lane bridge environment
    # inputs:
        # num agents (int > 1)
        # agent params - dictionary of agent parameters (e,a,g,method, mod)
    def __init__(self, num_agents,agent_params, random_seed):
        
        self.random_seed = random_seed
        self.size = num_agents + 3
        num_states = num_agents**self.size
        num_actions = 2
        num_agent_states = 27*self.size
        
        self.state = []
        self.agents = []
        self.num_agents = num_agents
        for i in range (0,num_agents):
            agent = {}
            agent['sa_vals'] = -20 + np.random.rand(num_agent_states,num_actions)*-20
            self.random_seed = self.random_seed + 1
            agent['params'] = agent_params
            self.agents.append(agent)

        
    #_________________________get_start_state()______________________________#
    # initializes the start state of the environment 
    # half of cars on one side of bridge, half on the other, 1 terminal position on either end of queues
    # example:
    # indices:    0  1 2 3 4 5 6 7 8
    # num lanes: inf 2 2 2 1 2 2 2 inf
    # agent locs: _  1 3 5 _ 4 2 0 _
    def get_start_state(self):
        self.state = np.zeros([self.num_agents])
        for i in range(0,self.num_agents):
            if i % 2 == 1: # odd agent
                self.state[i] = (i-1)/2+1 
            else: # even agent
                self.state[i] = self.size - (2+(i/2))
        return self.state


    #_________________________get_next_state()______________________________#
    # sets self.state equal to next state based on agents' actions and random prey move
    # inputs:
        # actions - a 1 x num_agents list where entry i corresponds to the integer
            # move for agent i
    def get_next_state(self, actions): 
        # move 0 = stay, 1 = advance
        for i in range(0,self.num_agents):
            # move even agents down unless in terminal state
            if i%2 == 0: #even agents
                if actions[i] == 1 and self.state[i] != 0:
                    self.state[i] = self.state[i] - 1
            # move odd agents up unless in terminal state
            else:
                if actions[i] == 1 and self.state[i] != self.size-1:
                    self.state[i] = self.state[i] + 1

        
                
                
    #____________________________get_reward()_________________________________#
    # gets reward for a state (first checks for agent collisions or prey capture)
    # returns: reward (integer)
    def get_reward(self):
        # get lists of even and odd locations
        evens = []
        odds = []
        terminals = 0
        # check for rear-ends and count terminal cars
        for i in range(0,self.num_agents):
            # even agents
            if i % 2 == 0:
                loc = self.state[i]
                if loc in evens:
                    # collision with other even ie. rear-end
                    reward = -10
                    break
                elif loc == 0:
                    # agent has reached terminal location
                    terminals = terminals + 1
                else:
                    evens.append(loc)
            else:
                loc = self.state[i]
                if loc in odds:
                    # collision with other even ie. rear-end
                    reward = -10
                    break
                elif loc == self.size-1:
                    # agent has reached terminal location
                    terminals = terminals + 1
                else:
                    odds.append(loc)
            
        # check whether evens and odds collided in 1-lane location
        center = (self.size-1)/2 
        if center in evens and center in odds:
            reward = -10
        
        # check whether all agents are in terminal states
        elif terminals == self.num_agents:
            reward = 1
            
        # penalize slightly to encourage speed
        else:
            reward = 0
            
        return reward
    
    #_________________________action_selection ()______________________________#
    # selects an action for one agent according to epsilon-greedy policy
    # inputs: agent_num (int)
    # returns: move (int 0 or 1)
    def action_selection(self,agent_num): #Q-learning and variants, SARSA and variants supported
        
        epsilon = self.agents[agent_num]['params']['epsilon']
        sa_vals = self.agents[agent_num]['sa_vals']
        state_num = self.agent_state(agent_num)
        
        # if rand < epsilon select random move
        rand = random.random()
        if rand < epsilon:
            move = random.randint(0,1)
            
        # else select best move
        else: 
            max_val = -1* np.inf
            max_move = 0
            vals = [0,1]
            random.shuffle(vals)
            for i in vals:
                if sa_vals[state_num,i] > max_val:
                    max_val = sa_vals[state_num,i]
                    max_move = i
            move = max_move
        
        return move, state_num
        
    #____________________________update_values()______________________________#
    # compare previously predicted value to new estimate of value and performs 
    # update simultaneously for all agents
    # inputs:
        # moves - 1 x num_agents list of moves (ints from 0-4) 
        # reward - the reward generated by moving to the current state
        # prev_state_num - unique tag specifying previous state (int)
    def update_values(self, moves, reward, prev_state_nums):
        
        next_moves = []
        # for one agent
        for i in range(0, self.num_agents):
            
            # store agent parameters in temp variables
            epsilon = self.agents[i]['params']['epsilon']
            alpha = self.agents[i]['params']['alpha']
            gamma = self.agents[i]['params']['gamma']
            method = self.agents[i]['params']['method'] # Q 
            mod = self.agents[i]['params']['mod'] # None, distributed or hysteretic
            
            state_num = self.agent_state(i)
            sa_vals = self.agents[i]['sa_vals']
            
            # select next move
            rand = random.random()
            # use current epsilon-greedy policy if SARSA
            if method == 'SARSA' and rand < epsilon:
                    next_move = random.randint(0,1)       
            # else select best move
            else:
                max_val = -1* np.inf
                max_move = 0
                vals = [0,1]
                random.shuffle(vals)
                for j in vals:
                    if sa_vals[state_num,j] > max_val:
                        max_val = sa_vals[state_num,j]
                        max_move = j
                next_move = max_move
                
            #get update sa_val for agents
            cur_state_num = self.agent_state(i)
            update_val = reward + gamma* sa_vals[cur_state_num,next_move]
            
            # deal with terminal states
            if reward == 1 or -10:
                update_val = reward
            
            # get difference between old and new values
            prev_val = sa_vals[prev_state_nums[i],moves[i]]
            diff = update_val - prev_val
            
            # add distributed and hysteretic functionality here
            if mod == 'distributed':
                if diff < 0:
                    alpha = 0
            elif mod == 'hysteretic':
                if diff < 0:
                    alpha = 0.1 * alpha
                    
            self.agents[i]['sa_vals'][prev_state_nums[i],moves[i]] = (1-alpha)*prev_val + alpha*(update_val)
            next_moves.append(next_move)
        return next_moves
                   
    #____________________________state_parser()_______________________________#
    # returns unique tag specifying current self.state (int) 
    def state_parser(self):
        num = 0
        for i in range(0,len(self.state)):
            num = num + int(self.state[i]) ** (1+i)
        return int(num)
    
    #_______________________________agent_state()_____________________________#
    # returns the percieved state by agent (current location and 3 spaces head)
    # inputs: agent_num (int)
    def agent_state(self,agent_num):
        evens = []
        odds = []
        for i in range(0,self.num_agents):
            # even agents
            if i % 2 == 0:
                loc = self.state[i]
                evens.append(loc)
            # odd agents
            else:
                loc = self.state[i]
                odds.append(loc)
        
        loc = self.state[agent_num]
        evens = []
        odds = []
        
        ahead1 = 0
        ahead2 = 0
        ahead3 = 0
        if agent_num % 2 == 0:
            if loc - 1 > 0:
                if (loc-1) in evens:
                    ahead1 = 2
                elif (loc-1) in odds:
                    ahead1 = 1
            if loc - 2 > 0:
                if (loc-2) in evens:
                    ahead1 = 2
                elif (loc-2) in odds:
                    ahead1 = 1        
            if loc - 3 > 0:
                if (loc-3) in evens:
                    ahead1 = 2
                elif (loc-3) in odds:
                    ahead1 = 1
        else:
            if loc + 1 < self.size-1:
                if (loc+1) in evens:
                    ahead1 = 2
                elif (loc+1) in odds:
                    ahead1 = 1
            if loc + 2 < self.size-1:
                if (loc+2) in evens:
                    ahead1 = 2
                elif (loc+2) in odds:
                    ahead1 = 1        
            if loc + 3 < self.size-1:
                if (loc+3) in evens:
                    ahead1 = 2
                elif (loc+3) in odds:
                    ahead1 = 1 
        # hash into unique value
        agent_percieved_state = int(loc+ahead1*10+ahead2*100+ahead3*1000)
        return agent_percieved_state
    
    #____________________________show_state()______________________________#
    # prints numerical representation of current state, where 2 represents a
    # predator and 1 represents prey, all other cells contain 0
    def show_state(self):
        stategrid = np.zeros([self.size])
        print("Locations of agents:")
        print(self.state)
        # add agents to grid
        for i in range(0,self.num_agents):
            loc = int(self.state[i])
            stategrid[loc] = stategrid[loc] +1
        print("Number of agents per location:")
        print(stategrid)
     