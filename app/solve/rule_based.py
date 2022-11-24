import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import os
import glob
import json
import random
import logging

from environment import Reward_Network

##### Rule_Agent class ######
#############################
class Rule_Agent:
    
    def __init__(self,networks,strategy,networks_filename):
        """
        Initializes a Rule Agent object, that follows a specified strategy

        Args:
            networks (list): list of Reward_Network objects
            strategy (str): solving strategy name 
            networks_filename (str): name fo the json file containing the networks
        """

        # assert tests
        assert strategy in ['highest_payoff','take_first_loss','random'], \
            f'Strategy name must be one of {["highest_payoff","take_first_loss","random"]}, got {strategy}'
        
        self.networks = networks
        self.strategy =  strategy        
        self.networks_filename = networks_filename

        # colors for plot
        self.colors = {'highest_payoff':'skyblue','take_first_loss':'orangered','random':'springgreen'}
        

    def select_action(self,possible_actions,possible_actions_rewards):
        """
        We are in a current state S. Given the possible actions from S and the rewards
        associated to them this method returns the action to select (based on the current 
        solving strategy)

        Args:
            possible_actions (np.array): array containing next possible states (expressed with node numbers) 
            possible_actions_rewards (np.array): array containing rewards of next possible states

        Returns:
            (np.array): selected action
        """

        if self.strategy == 'take_first_loss':
            print(self.strategy,self.loss_counter,possible_actions_rewards)

        if self.strategy=='random':
            return random.choice(possible_actions)

        # take first loss -> select among possible actions the one that gives best reward BUT make sure to take a first big loss
        if self.strategy == 'take_first_loss' and self.loss_counter<1 and -100 in possible_actions_rewards:
            self.loss_counter +=1

            if len(np.argwhere(possible_actions_rewards==-100)[0])!=2: # that is, we have only one big loss in the possible actions
                return possible_actions[np.argwhere(possible_actions_rewards==-100)[0][0]]
            else: # else if both actions lead to big loss pick a random one
                return possible_actions[random.choice(np.argwhere(possible_actions_rewards==-100)[0])]
        else:

            try:
                if not np.all(possible_actions_rewards == possible_actions_rewards[0]):
                    return possible_actions[np.argmax(possible_actions_rewards)]
                else:
                    return random.choice(possible_actions)
            except:
                print(f'Error in network {self.environment.id}')
                print(self.environment.action_space)

    def solve(self):
        """
        Ths method solves the given networks, with different constraints depending on the strategy.
        Returns solution in tabular form

        Args:
            network (Reward_Network object): a network with info on nodes,edges
        """        

        for network in self.networks:

            if self.strategy == 'take_first_loss':
                self.loss_counter = 0 # to reset!

            # solution variables
            self.solution = []   
            self.solutions = [] 
            self.all_solutions_filename = os.path.join(solutions_dir,f'{self.strategy}_{self.networks_filename}.csv')
            self.solution_columns = ["network_id", "strategy", "step", "source_node", "current_node", "reward", "total_reward"]

            # network environment variables
            self.environment = Reward_Network(network)
            self.environment.reset()
            
            while self.environment.is_done==False:
                s = []
                obs = self.environment.observe()
                a = self.select_action(obs['actions_available'],obs['next_possible_rewards'])
                step = self.environment.step(a)
                s.append(self.environment.id)
                s.append(self.strategy)
                s.append(step['n_steps'])
                s.append(step['source_node'])
                s.append(step['current_node'])
                s.append(step['reward'])
                s.append(step['total_reward'])
                self.solution.append(s)
            print('\n')
            self.solution_df = pd.DataFrame(self.solution, columns = self.solution_columns)
            self.solutions.append(self.solution_df)

        self.df = pd.concat(self.solutions,ignore_index=True)
        self.df[self.solution_columns].to_csv(self.all_solutions_filename,sep='\t')


    def inspect_solutions(self):
        """
        this method creates a plot of final rewards' distribution
        """
        self.df = pd.read_csv(os.path.join(solutions_dir,f'{self.strategy}_train_viz_100.csv'),sep='\t')
        g = sns.displot(data=self.df[self.df['step']==8], x="total_reward", kde=True, color=self.colors[self.strategy])
        g.set(xlim=(-400,400),xlabel='Final reward',ylabel='Count')
        plt.show()

    def save_solutions_frontend(self):
        """
        This method saves the selected strategy solution of the networks to be used in the experiment frontend;
        solutions are saved in a JSON file with network id and associated list of moves
        """
        assert os.path.exists(os.path.join(solutions_dir,f'{self.strategy}_{self.networks_filename}.csv')),f'solutions file not found!'
        
        df = pd.read_csv(os.path.join(solutions_dir,f'{self.strategy}_{self.networks_filename}.csv'),sep='\t')

        def add_source(x):
            a=x
            a.insert(0,0)
            return a

        s = df.groupby(['network_id'])['current_node'].apply(list).reset_index(name='moves')
        s['moves'] = s['moves'].apply(add_source)
        obj = s.to_dict('records')

        with open(os.path.join(solutions_dir,f'solution_moves_{self.strategy}_viz.json'), 'w') as f:
            json.dump(obj, f)