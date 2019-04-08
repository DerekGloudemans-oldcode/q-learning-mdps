# environment_pred
# utility functions for cooperative MDPs with Q-learning and SARSA

# import required packages
import numpy as np
import random



########################## Environment_pred Class ##############################
# Environment classes provide functions for maintaining agents, the environment, 
# and the interaction between them
# Environment_pred represents the predator CMDP as discussed in Laetitia et al.
class Environment_pred():
    
    #_________________________Environment_pred()______________________________#
    # initializes a predator game environment
    # inputs:
        # num agents (int)
        # size grid for game is size x size (int)
        # agent params - dictionary of agent parameters (e,a,g,method, mod)
    def __init__(self, num_agents,size,agent_params,random_seed):
        
        num_states = size**(num_agents*2+2)
        num_actions = 5
        self.random_seed = random_seed
        self.state = []
        self.size = size
        self.agents = []
        self.num_agents = num_agents
        for i in range (0,num_agents):
            agent = {}
            agent['sa_vals'] = -20 + np.random.rand(num_states,num_actions)*-10
            self.random_seed = self.random_seed+1
            agent['params'] = agent_params
            self.agents.append(agent)

        
    #_________________________get_start_state()______________________________#
    # initializes the start state of the environment randomly
    def get_start_state(self):
        self.state = np.zeros([self.num_agents+1,2])
        for i in range(0,len(self.state)):
            avail = False
            occupied = []
            while not avail:
                x = random.randint(0,self.size-1)
                y = random.randint(0,self.size-1)
                if (x,y) not in occupied:
                    occupied.append( (x,y))
                    self.state[i,0] = y
                    self.state[i,1] = x
                    avail = True
        return self.state


    #_________________________get_next_state()______________________________#
    # sets self.state equal to next state based on agents' actions and random prey move
    # inputs:
        # actions - a 1 x num_agents list where entry i corresponds to the integer
            # move for agent i
    def get_next_state(self, actions): 
        # find possible moves for prey
        agent_occupieds = [self.state[i,0]*self.size+self.state[i,1] for i in range(0,len(self.state)-1)]
        poss_moves = []
        if ((self.state[-1,0]+1)*self.size+self.state[-1,1]) not in agent_occupieds and self.state[-1,0]+1 < self.size:
            poss_moves.append(2)
        if  ((self.state[-1,0]-1)*self.size+self.state[-1,1]) not in agent_occupieds and self.state[-1,0]-1 >= 0: 
            poss_moves.append(0)  
        if ((self.state[-1,1]+1)*self.size+self.state[-1,1]) not in agent_occupieds and self.state[-1,1]+1 < self.size:
            poss_moves.append(1)
        if  ((self.state[-1,1]-1)*self.size+self.state[-1,1]) not in agent_occupieds and self.state[-1,1]-1 >= 0:
            poss_moves.append(3) 
               
        # move agents 
        for i in range(0,self.num_agents):
            if actions[i] == 1:
                if self.state[i,1] < self.size-1:
                    self.state[i,1] = self.state[i,1] + 1
            elif actions[i] == 2:
                if self.state[i,0] < self.size-1:
                    self.state[i,0] = self.state[i,0] + 1        
            elif actions[i] == 3:
                if self.state[i,1] > 0:
                    self.state[i,1] = self.state[i,1] - 1 
            elif actions[i] == 0:
                if self.state[i,0] > 0:
                    self.state[i,0] = self.state[i,0] - 1
        
        # move prey
        stay = random.random()
        if stay > 0.2:
            if len(poss_moves) > 0:
                move = poss_moves[random.randint(0,len(poss_moves)-1)]
                if move == 0:
                    self.state[-1,0] = self.state[-1,0]-1
                elif move ==1:
                    self.state[-1,1] = self.state[-1,1]+1
                elif move == 2:
                    self.state[-1,0] = self.state[-1,0]+1
                elif move == 3:
                    self.state[-1,1] = self.state[-1,1]-1
                
                
    #____________________________get_reward()_________________________________#
    # gets reward for a state (first checks for agent collisions or prey capture)
    # returns: reward (integer)
    def get_reward(self):
        # check if any agents collided
        agent_pos = []
        for i in range(0,len(self.state)-1):
            agent_pos.append(self.state[i,0]*self.size+self.state[i,1])
        if len(set(agent_pos)) != len(agent_pos):
            # collision has occurred
            reward = -0.1
            #reposition all agents
            for i in range(0,len(self.state-1)):
                self.state[i,0] = random.randint(0,self.size-1)
                self.state[i,1] = random.randint(0,self.size-1)
        # check if any agents captured the prey
        elif self.state[-1,0]*self.size+self.state[-1,1] in agent_pos:
            reward = 1
        else:
            reward = 0
            
        return reward
    
    #_________________________action_selection ()______________________________#
    # selects an action for one agent according to epsilon-greedy policy
    # inputs: agent_num (int)
    # returns: move (int from 0-4)
    def action_selection(self,agent_num): #Q-learning and variants, SARSA and variants supported
        
        epsilon = self.agents[agent_num]['params']['epsilon']
        sa_vals = self.agents[agent_num]['sa_vals']

        # if rand < epsilon select random move
        rand = random.random()
        if rand < epsilon:
            move = random.randint(0,4)
            
        # else select best move
        else:
            state_num = self.state_parser()
            max_val = -1* np.inf
            max_move = 0
            vals = [i for i in range(0,5)]
            random.shuffle(vals)
            for i in vals:
                if sa_vals[state_num,i] > max_val:
                    max_val = sa_vals[state_num,i]
                    max_move = i
            move = max_move
        
        return move, 1
        
    #____________________________update_values()______________________________#
    # compare previously predicted value to new estimate of value and performs 
    # update simultaneously for all agents
    # inputs:
        # moves - 1 x num_agents list of moves (ints from 0-4) 
        # reward - the reward generated by moving to the current state
        # prev_state_num - unique tag specifying previous state (int)
    def update_values(self, moves, reward, prev_state_num):
        next_moves = []
        # for one agent
        for i in range(0, self.num_agents):
            
            # store agent parameters in temp variables
            epsilon = self.agents[i]['params']['epsilon']
            alpha = self.agents[i]['params']['alpha']
            gamma = self.agents[i]['params']['gamma']
            method = self.agents[i]['params']['method'] # Q 
            mod = self.agents[i]['params']['mod'] # None, distributed or hysteretic
            
            state_num = self.state_parser()
            sa_vals = self.agents[i]['sa_vals']
            
            # select next move
            rand = random.random()
            # use current epsilon-greedy policy if SARSA
            if method == 'SARSA' and rand < epsilon:
                    next_move = random.randint(0,4)       
            # else select best move
            else:
                state_num = self.state_parser()
                max_val = -1* np.inf
                max_move = 0
                vals = [0,1,2,3,4]
                random.shuffle(vals)
                for j in vals:
                    if sa_vals[state_num,j] > max_val:
                        max_val = sa_vals[state_num,j]
                        max_move = j
                next_move = max_move
                
            #get update sa_val for agents
            cur_state_num = self.state_parser()
            update_val = reward + sa_vals[cur_state_num,next_move]
            
            # deal with terminal states
            if reward == 1 or -0.1:
                update_val = reward
            
            # get difference between old and new values
            prev_val = sa_vals[prev_state_num,moves[i]]
            diff = update_val - prev_val
            
            # add distributed and hysteretic functionality here
            if mod == 'distributed':
                if diff < 0:
                    alpha = 0
            elif mod == 'hysteretic':
                if diff < 0:
                    alpha = 0.1 * alpha
                    
            self.agents[i]['sa_vals'][prev_state_num,moves[i]] = alpha*(prev_val + gamma*(diff))
            next_moves.append(next_move)
        return next_moves
                   
    #____________________________state_parser()______________________________#
    # returns unique tag specifying current self.state (int) 
    def state_parser(self):
        num = 0
        for i in range(0,len(self.state)):
            num = num + (int(self.state[i,0]*self.size)+int(self.state[i,1])) \
            *int(self.size**(2*i))
        return int(num)
    
    
    #____________________________show_state()______________________________#
    # prints numerical representation of current state, where 2 represents a
    # predator and 1 represents prey, all other cells contain 0
    def show_state(self):
        stategrid = np.zeros([self.size,self.size])
        print(self.state)
        # add agents to grid
        for i in range(0,len(self.state)-1):
            stategrid[int(self.state[i,0]),int(self.state[i,1])] = 2
        # add prey to grid
        stategrid[int(self.state[-1,0]),int(self.state[-1,1])] = 1
        
        print(stategrid)