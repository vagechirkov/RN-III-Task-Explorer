import json
import yaml
import sys
import shutil
import os
import torch as th
import torch_scatter


# basics


def save_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def save_yaml(obj, filename):
    with open(filename, 'w') as f:
        yaml.dump(obj, f)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


def get_subfolders(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]


def get_subfiles(d):
    return [o for o in os.listdir(d)]


def make_dir(directory, clean=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif clean:
        shutil.rmtree(directory)
        os.makedirs(directory)


# utlitiy functions (network)
def parse_network(network):
    """
        Calculate the reward and transition matrices.
        R (original node : action):
            reward on transition from original node by doing the action
        T (original node : desitination node : action):
            one if action from original node leads to destination node, 0 otherwise

    """
    actions = network['edges']
    edges = th.tensor(
        [
            [action['source_num'], action['target_num']]
            for action in actions
        ]
    )
    rewards = th.tensor(
        [
            action['reward']
            for action in actions
        ]
    )
    return edges, rewards


def groupby_transform_max(src, index, dim_size):
    """
    Maximise values in src across index, while keeping the shape of src.
    """
    max, argmax = torch_scatter.scatter_max(
        src, index, dim=-1, dim_size=dim_size)
    return max[index]


def calculate_q_value(edges, rewards, n_steps, n_nodes, gamma):
    """
        Q [step, edge]: estimated action value for step edge pairs
    """
    Q = th.zeros((n_steps+1, len(edges)))  # (step, edge)

    for k in range(n_steps-1, -1, -1):
        # max next action value for each source node
        max_next_value = torch_scatter.scatter_max(
            Q[k+1], edges[:, 0], dim=-1, dim_size=n_nodes)[0]  # (node)
        # projecting max next action values on edge target
        max_next_value = max_next_value[edges[:, 1]]  # (edge)
        # calculating Q value
        Q[k] = (1-gamma) * max_next_value + rewards
    return Q[:-1]


def calculate_trace(Q, edges, starting_node):
    """
        Calculating traces of edges, nodes and rewards for a given starting node.
    """
    n_steps = Q.shape[0]
    node_trace = th.empty(n_steps, dtype=th.int64)
    edge_trace = th.empty(n_steps, dtype=th.int64)
    current_node = starting_node
    node_trace
    for s in range(n_steps):
        edge_idx = th.where(
            current_node == edges[:, 0], Q[s], th.tensor(-10000.)).argmax()
        current_node = edges[edge_idx, 1]
        edge_trace[s] = edge_idx
        node_trace[s] = current_node
    return edge_trace, node_trace
