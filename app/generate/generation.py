# generation
##############################

import networkx as nx
import string
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from .utils import save_json, make_dir
import pydantic
import hashlib
import json
from collections import Counter
from .utils import parse_network, calculate_q_value, calculate_trace
import torch as th
import torch_scatter
from typing import Optional,List,Dict,Any
from models.network import Network
from pydantic import BaseModel,validator,parse_obj_as,ValidationError,root_validator
import yaml
import sys
import shutil
import streamlit as st
import itertools


seed = 42
random.seed(seed)
np.random.seed(seed)

##### Network_Generator class ######
####################################
class Network_Generator:

    def __init__(self,params):
        """
        Initializes a network generator object with parameters
        obtained from streamlit form

        Args:
            params (dict): parameter dictionary
        """

        self.n_rewards = params['n_rewards']
        self.rewards = params['rewards']
        self.n_networks = params['n_networks']
        self.n_steps = params['n_steps']
        self.n_levels = params['n_levels']
        self.node_stages = {0: 4,
                            1: 1,
                            2: 1,
                            3: 1}

        # parameters for visualization
        self.node_size=2200
        self.arc_rad = 0.1

        # colors
        self.colors = ["red", "coral", "lightgrey", "lightgreen", "green"]
        self.stage_colors=['paleturquoise','lightskyblue','royalblue','blue']
        min_al = 3
        c = itertools.product(list(self.node_stages.keys()), repeat=2)
        

        # TODO change form_to_str depending on n_rewards and n_levels
        from_to_str = {
            "(0,0)": [1, 2, 3],
            "(0,1)": [0],
            "(1,0)": [1, 2],
            "(1,1)": [2, 3],
            "(1,2)": [2, 3],
            "(2,0)": [1, 2],
            "(2,1)": [1, 2],
            "(2,2)": [2, 3],
            "(2,3)": [2, 3],
            "(3,0)": [3, 4],
            "(3,1)": [3, 4],
            "(3,2)": [3, 4],
            "(3,3)": [3, 4],
        }
        # idx = np.array_split(np.arange(0,self.n_rewards,1),self.n_levels)
        # from_to_str = {
        #     "(0,0)": idx[random.randint(0,self.n_rewards)],
        #     "(0,1)": idx[random.randint(0,self.n_rewards)],
        #     "(1,0)": idx[random.randint(0,self.n_rewards)],
        #     "(1,1)": idx[random.randint(0,self.n_rewards)],
        #     "(1,2)": idx[random.randint(0,self.n_rewards)],
        #     "(2,0)": idx[random.randint(0,self.n_rewards)],
        #     "(2,1)": idx[random.randint(0,self.n_rewards)],
        #     "(2,2)": idx[random.randint(0,self.n_rewards)],
        #     "(2,3)": idx[random.randint(0,self.n_rewards)],
        #     "(3,0)": idx[random.randint(0,self.n_rewards)],
        #     "(3,1)": idx[random.randint(0,self.n_rewards)],
        #     "(3,2)": idx[random.randint(0,self.n_rewards)],
        #     "(3,3)": idx[random.randint(0,self.n_rewards)],
        # }
        # yaml does not support objects with tuples as keys. This is a hacky workaround.
        self.from_to = {(int(k[1]),int(k[3])):v for k,v in from_to_str.items()}




##### directory and filenames specification ######
##################################################
# output_folder= '../../data/rawdata'
# pydantic_folder= '../models'
# networks_file = os.path.join(output_folder, 'networks.json')
# make_dir(output_folder)



    # individual network building functions
    #######################################
    def add_link(self,G, a, b):
        a_s = G.nodes[a]['stage']
        b_s = G.nodes[b]['stage']
        reward_idx = random.choice(self.from_to[(a_s,b_s)])
        G.add_edge(a, b, reward=self.rewards[reward_idx], reward_idx=reward_idx, color=self.colors[reward_idx])

    def new_node(self,G, stage):
        idx = len(G)
        name = string.ascii_uppercase[idx%len(string.ascii_lowercase)]
        G.add_node(idx, name=name, stage=stage)
        return idx

    def filter_node(self,G, node, current_node):
        if node == current_node:
            return []
        if node in G[current_node]:
            return []
        a_s = G.nodes[current_node]['stage']
        b_s = G.nodes[node]['stage']
        return self.from_to.get((a_s, b_s), [])


    def get_min_incoming(self,G, nodes):
        in_de = [G.in_degree(n) for n in nodes]
        min_in_de = min(in_de)
        return [n for n, ind in zip(nodes, in_de) if ind == min_in_de]

        
    def find_nodes(self,G, current_node):
        return [
            n
            for n in G.nodes() 
            for _ in self.filter_node(G, n, current_node)
        ]


    def sample_network(self):
        G = nx.DiGraph()

        # node_stages = {
        #     0: 4,
        #     1: 1,
        #     2: 1,
        #     3: 1
        # }

        # assigns a new node to a random stage, a total of 3 times
        for i in range(3):
            stage = random.randint(0,3)
            self.node_stages[stage] += 1
        
        # assigns node to stage
        for stage, n in self.node_stages.items():
            for i in range(n):
                self.new_node(G, stage)

        start_node = 0
        n_edges_per_node = 2

        for i in range(60):
            current_node = start_node
            for j in range(8):
                if len(G.out_edges(current_node)) < n_edges_per_node:
                    potential_nodes = self.find_nodes(G, current_node)
                    potential_nodes = self.get_min_incoming(G, potential_nodes)
                    next_node = random.choice(potential_nodes)
                    self.add_link(G, current_node, next_node)
                    if len(G.edges) == (len(G.nodes) * n_edges_per_node):
                        break
                else:
                    next_node = random.choice([e[1] for e in G.out_edges(current_node)])
                current_node = next_node
        return G


    ####### parsing functions ########
    ####################################
    def parse_node(self,name, pos_map, id, **kwargs):
        return {
            'node_num': id,
            'display_name': name,
            'node_size': 3,
            'level':kwargs['stage'],
            **pos_map[id]
        }


    def parse_link(self,source, target, reward, reward_idx,arc_type,source_x,source_y,arc_x,arc_y,target_x,target_y, **_):
        return {
            "source_num": source,
            "target_num": target,
            "reward": reward,
            "arc_type": arc_type,
            "source_x": source_x,
            "source_y": source_y,
            "arc_x": arc_x,
            "arc_y": arc_y,
            "target_x": target_x,
            "target_y": target_y

        }


    def create_base_network_object(self,pos_map, starting_node=0, *, nodes, links, network_id, n_steps, **kwargs):
        return {
            'network_id': network_id,
            'nodes': [self.parse_node(pos_map=pos_map, **n) for n in nodes],
            'edges': [self.parse_link(**l) for l in links],
            'starting_node': starting_node}


    def get_max_reward(self,network):
        edges, rewards = parse_network(network)
        # calculate q value for gamma = 0 (no pruning)
        Q = calculate_q_value(edges, rewards, n_steps=self.n_steps, n_nodes=len(network['nodes']), gamma=0)
        # get trace corresponding to q values
        edge_trace, node_trace = calculate_trace(Q, edges, starting_node=network['starting_node'])
        reward_trace = rewards[edge_trace]
        max_reward = reward_trace.sum()

        max_reward2 = th.where(network['starting_node'] == edges[:, 0], Q[0], th.tensor(-10000.)).max()

        if max_reward2 != max_reward:
            print('Rewards do not match')
            print(network['network_id'], max_reward2, max_reward)

        # TODO: check why this assertion is not always valid.
        # assert max_reward == max_reward2
        return max_reward2
        

    def add_max_reward(self,network):
        return {
            **network,
            'max_reward': self.get_max_reward(network).item()
        }

    def create_network_object(self,**kwargs):
        network = self.create_base_network_object(**kwargs)
        return self.add_max_reward(network)



    ##### generation and storing networks in json file ######
    #########################################################
    #@st.cache(allow_reuse=False)
    def generate(self,save_path=None):
        """
        Using the functions defined above this method generates networks with
        visualization info included. 
        The generated network(s) are also saved in a json file at location
        specified by save_path

        Args:
            save_path (str): path to location where to store generated networks JSON
        """

        # sample and store training networks
        self.networks = []
        for i in range(self.n_networks):
        
            G = self.sample_network()
            N = nx.json_graph.node_link_data(G)

            # NEW: shuffle randomly the order of the nodes in circular layout
            pos = nx.circular_layout(G)
            node_order = list(G.nodes(data=True))
            random.shuffle(node_order)
            random_pos = {}
            for a,b in zip(node_order, [pos[node] for node in G]):
                random_pos[a[0]] = b

            pos_map = { k: {'x': v[0]*100, 'y': v[1]*-1*100}
                        for k, v in random_pos.items()}


            # NEW: add vertices for visualization purposes
            for ii, e in enumerate(G.edges()):
                if (reversed(e) in G.edges()):
                    N['links'][ii]['arc_type'] = 'curved'
                    arc = nx.draw_networkx_edges( 
                        G, random_pos, edgelist=[e], node_size=self.node_size,
                        connectionstyle=f'arc3, rad = {self.arc_rad}')
                else:
                    N['links'][ii]['arc_type'] = 'straight'
                    arc = nx.draw_networkx_edges(
                        G, random_pos, edgelist=[e], node_size=self.node_size)
                
                vert = arc[0].get_path().vertices.T[:, :3] * 100

                N['links'][ii]['source_x'] = vert[0, 0] 
                N['links'][ii]['source_y'] = -1* vert[1, 0]
                N['links'][ii]['arc_x'] = vert[0, 1]
                N['links'][ii]['arc_y'] = -1*vert[1, 1]
                N['links'][ii]['target_x'] = vert[0, 2]
                N['links'][ii]['target_y'] = -1* vert[1, 2]

            network_id = hashlib.md5(json.dumps(N, sort_keys=True).encode('utf-8')).hexdigest()

            c = Counter([e['source'] for e in N['links']])
            
            if all(value == 2 for value in c.values()) and len(list(c.keys()))==10:
                create_network = self.create_network_object(pos_map=pos_map, 
                                                            n_steps=self.n_steps, 
                                                            network_id=network_id, 
                                                            **N)
                self.networks.append(create_network)
            else:
                print(f'counter {c}, nodes are {list(c.keys())} (n={len(list(c.keys()))})')

        #st.json(networks[0])
        #save_json(networks, save_path)
        network_objects = [Network(**n) for n in self.networks]
        #st.json(network_objects[0].json())
        return self.networks # network_objects

    def save_as_json(self):
        return json.dumps(self.networks)