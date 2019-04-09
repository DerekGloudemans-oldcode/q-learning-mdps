# utilfn-CMDP-QL
# utility functions for cooperative MDPs with Q-learning and SARSA

# import required packages
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import itertools
import random
import _pickle as pickle
from environment_pred import Environment_pred
from environment_lane import Environment_lane
from environment_mat import Environment_mat

#configure plot properties
plt.style.use('seaborn')
plt.rcParams.update({'font.size': 20})

############################### Simulator Class ###############################
# Runs episodes (same environement with no reinitialization), 
# simulations (same environment with reinitialization), and
# trials (new agent parameters)
# also contains result-plotting functions
# all_trial_params - dictionary, contains e,g,a,learner,modifications
class Simulator():
    
    def __init__(self,all_params,n_sims,n_eps,n_agents,env,multip):
        self.all_params = all_params #dictionary
        self.num_episodes = n_eps
        self.num_sims = n_sims
        self.num_agents = n_agents
        self.env_type = env
        self.multip = multip
        self.results = []
      
        
    #_____________________________trial()_____________________________________
    # Runs and averages a number of simulations using the same parameters 
    # inputs:
        # params - dictionary of agent parameters
        # results_queue - multiprocessing Queue to which results are written
            # Nonetype object passed if multiprocessing disabled
    # returns 2-tuple:
        # params - as above, for tagging trial
        # sum_all_rewards - sum of rewards per episode averaged over simulations (1-d numpy array)
    def trial(self,params,results_queue):
        random.seed = 1
        sum_all_rewards = np.zeros(self.num_episodes)
        sum_all_steps   = np.zeros(self.num_episodes)
        
        for i in range(0,self.num_sims):
            if self.multip:
                results_queue.put("On simulation {} of {}".format(i+1,self.num_sims))
            else:
                print("On simulation {} of {}".format(i+1,self.num_sims))
            random_seed = (i+1)*self.num_sims*self.num_agents
            
            rewards,steps = self.simulation(params,random_seed)
            sum_all_rewards = sum_all_rewards + rewards
            sum_all_steps = steps + sum_all_steps
            
        sum_all_rewards = sum_all_rewards/float(self.num_sims)
        sum_all_steps   = sum_all_steps/float(self.num_sims)
        if self.multip:
            results_queue.put((params,sum_all_rewards,sum_all_steps))
        return (params,sum_all_rewards,sum_all_steps)
    
    
    #____________________________simulation()_________________________________#
    # Runs a number of episodes using the same environment and continually updated
    # state-action values
    # inputs:
        # params - dictionary of agent parameters
        # random_seed unique seed for each simulation in a trial (int)
    # returns:
        # all-avg_rewards - sum of rewards per episode (1-d numpy array)
        # sum_all_rewards - env (the Environment object with final SA vales)
    def simulation(self,params,random_seed):
        
        # initialize environment - this line changes based on Environment
        if self.env_type == 'lane':
            env = Environment_lane(self.num_agents,params,random_seed)
        elif self.env_type == 'pred':
            env = Environment_pred(self.num_agents,4,params,random_seed)
        elif self.env_type == 'mat':
            env = Environment_mat(True,params)
            
        # initialize arrays to hold results
        all_avg_rewards = np.zeros(self.num_episodes)
        all_steps = np.zeros(self.num_episodes)
        
        # get results for all episodes
        for i in range(0,self.num_episodes):          
            random_seed_new = random_seed*(1+i)
            all_avg_rewards[i], all_steps[i] = self.episode(env,random_seed_new,i)
            
        return all_avg_rewards,all_steps
    
    
    #_______________________________episode()_________________________________#
    # Runs one episode of the environment until a terminal state is reached
    # inputs: 
        # env -the environment in which episode runs
            # must have basic operational functionality to get start state, get
            # next state, select actions, get reward, and update SA values
            # as well as maintain agent SA vals and state representation
        # random_seed - unique for each episode (int)
        # episode_num (int)
    # returns: sum_rewards - sum of all rewards recieved by agents during episode
    def episode(self,env,random_seed,episode_num):
        show = False
        random.seed = random_seed
        steps = 0
        sum_rewards = 0
        
        # get start state and verify not terminal
        terminal = True
        while terminal:
            env.get_start_state()
            if env.get_reward() == 0 or self.env_type == 'mat':
                terminal = False
        
        # modify epsilon values if 'decreasing'
        if env.agents[0]['params']['epsilon'] == 'decreasing' and True:
            for a in env.agents:
                a['params']['epsilon'] = 0.05/(episode_num/10+1.0)
        
        #used for SARSA
        next_moves = []
        while not terminal:
                
            # select actions for all agents
            moves = []
            prev_state_nums = []
            
            
            # save move and, in the case of lane environemnt, previous percieved state
            for i in range(0,env.num_agents):
                move,prev_state_num = env.action_selection(i)
                # if SARSA, instead use the moves selected in the previous update_values call
                if env.agents[i]['params']['method'] == 'SARSA' and len(next_moves) > 0:
                    move = next_moves[i]
                moves.append(move)
                # append local state to prev_state_nums
                prev_state_nums.append(prev_state_num)
                
            # save state_num before advancing states -pred uses a global state 
            if self.env_type == 'pred':
                prev_state_nums = env.state_parser()
            
            # make actions - gets next state
            env.get_next_state(moves)
            if show:
                print("Moves: {}".format(moves))
                env.show_state()
                
            # get reward
            reward = env.get_reward()
            
            # update state-action vals
            next_moves = env.update_values(moves,reward,prev_state_nums)
            
            sum_rewards = sum_rewards + reward
            steps = steps + 1
            
            # if reward was goal state, terminate
            # for matrix games, all episodes only 1 step
            if reward == 1 or reward == -10 or self.env_type == 'mat':
                terminal = True
                errors = 0
                if reward == -10:
                    errors = 1
                    # arbitrarily increase step count if an error occurs
                    steps = steps + 40
                    
            # monitor current progress of episode        
            if show:   
                print("Sum rewards: {} N steps: {}".format(sum_rewards, steps))
        
        return sum_rewards, steps
    
    
    #____________________________run_all_trials()_________________________________#
    # Runs a trial for each unique combination of agent parameters in Simulator
    # returns: results - list of 2 tuples containing label for trial 
            # and trial average rewards per episode
    def run_all_trials(self):
        
        for epsilon,gamma,alpha,method,mod in itertools.product( \
                    self.all_params['epsilons'], \
                    self.all_params['gammas'], \
                    self.all_params['alphas'], \
                    self.all_params['methods'], \
                    self.all_params['mods']):
            params = {'alpha':        alpha,
                      'gamma':        gamma,
                      'epsilon':      epsilon,
                      'method':       method,
                      'mod':          mod
                      }
            result = self.trial(params,None)
            self.results.append(result)
        return self.results

#____________________________run_all_trials()_________________________________#
    # Runs a trial for each unique combination of agent parameters in parallel
    # inputs: n_trials - number of trials (int)
    # returns: results - list of 2 tuples containing label for trial 
            # and trial average rewards per episode    
    def run_all_trials_mp(self,n_trials):
        
        pool = mp.Pool(processes=6)
        m = mp.Manager()
        results_queue = m.Queue()
        for epsilon,gamma,alpha,method,mod in itertools.product( \
                    self.all_params['epsilons'], \
                    self.all_params['gammas'], \
                    self.all_params['alphas'], \
                    self.all_params['methods'], \
                    self.all_params['mods']):
            params = {'alpha':        alpha,
                      'gamma':        gamma,
                      'epsilon':      epsilon,
                      'method':       method,
                      'mod':          mod
                      }
            out = pool.apply_async(self.trial,(params,results_queue))
        #read all results from the results_queue into readable format
        while len(self.results) < n_trials :
            msg = results_queue.get()
            print(msg)
            if type(msg) == tuple:
                self.results.append(msg)
                print("{} processes terminated.".format(len(self.results)))
        
        print("All processes created. Waiting for processes to terminate.")
        while 1:
            if len(self.results) == n_trials:
                pool.close()
                pool.join()
                print("pool closed. everybody out of the water")
                break
        return self.results

    #____________________________plot_all_trials()_________________________________#
    # Smooths and plots results for all trials
    # inputs: smoothing - specifies size of window for smoothing 
    def plot_all_trials(self, smoothing = 11):  
        # parse data and labels
        results = self.results
        vals = []
        vals2 = []
        labels = []
        for i in range(0,len(results)):
            if results[i][0]['epsilon'] == 0.1 or True:
                vals.append(results[i][1])
                vals2.append(results[i][2])
                params = results[i][0]
                label = "{} {}, E:{}, A:{}, G:{}".format(params['mod'],params['method'],\
                         params['epsilon'],params['alpha'], params['gamma'])
                labels.append(label)
            
        # plot data
        plt.figure(figsize = (10,10))
        # smooth data
        for val in vals:
            plt.plot(self.hamming_smoother(val,smoothing))

        plt.legend(labels,prop={'size': 16})
        plt.ylabel("Average reward sum per episode",fontsize = 16)
        plt.xlabel("episodes",fontsize = 16)
        plt.title("Average reward per episode over {} simulations".format(self.num_sims),fontsize = 20)        
    
        plt.figure(figsize = (10,10))
        for val2 in vals2:
            plt.plot(self.hamming_smoother(val2,smoothing))
        plt.legend(labels,prop={'size': 16})
        plt.ylabel("Average number of steps per",fontsize = 16)
        plt.xlabel("episodes",fontsize = 16)
        plt.title("Average steps per episode over {} simulations".format(self.num_sims),fontsize = 20)
    
    
    #____________________________hamming_smoother()___________________________#
    # utility function for smoothing data
    # inputs:
        # unsmooth_data - (1d numpy array)
        # n_width - specifies amount of smoothing (odd int)
    #returns: ch -smoothed dat channel (1d numpy array)
    def hamming_smoother(self,unsmooth_data, n_width=11):
        ham = np.hamming(n_width)
        total = sum(ham)
        ham = ham/total
        ch = np.convolve(unsmooth_data, ham)
        return ch[int((n_width-1)):int(np.size(ch)-((n_width-1)))]
    
    
############################# BEGIN BODY CODE #################################
        
if __name__ == '__main__':

    # define agent parameters
    all_params = {
    'epsilons'    : ['decreasing'],
    'gammas'      : [0.9],
    'alphas'      : [0.4],
    'methods'     : ['Q'],
    'mods'        : ['','distributed','hysteretic']
    }
    
    # define parameters for simulator
    multip = False
    n_sims = 3000
    n_eps = 2000
    n_agents = 2
    env = 'mat'
    n_trials = len(all_params['epsilons'])*len(all_params['alphas'])* \
    len(all_params['gammas'])*len(all_params['methods'])*len(all_params['mods'])
    
    # create simulator object
    sim = Simulator(all_params,n_sims,n_eps,n_agents,env,multip)
    
    #run all trials
    if multip:
        results = sim.run_all_trials_mp(n_trials)
    else:
        results = sim.run_all_trials()
    vals = []
    
    # plot results
    sim.plot_all_trials(21)
    
    # save results
    f = open("results_temp_name.cpkl",'wb')
    pickle.dump(sim.results,f)
    f.close()
