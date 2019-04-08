from simulator import Simulator
import _pickle as pickle

f = open("final_results_lane_random_init_no_step_penalty.cpkl",'rb')
sim.results = pickle.load(f)
f.close()

sim.plot_all_trials(51)