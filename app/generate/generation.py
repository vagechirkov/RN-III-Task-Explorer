import hashlib
import json
import random
import string
from collections import Counter

import networkx as nx
import numpy as np

from models.network import Network
from models.environment import Environment

# from .utils import parse_network, calculate_q_value, calculate_trace

seed = 42
random.seed(seed)
np.random.seed(seed)


class NetworkGenerator:
    """
    Network Generator class
    """

    def __init__(self, environment: Environment):
        """
        Initializes a network generator object with parameters
        obtained from streamlit form

        Args:
            params (dict): parameter dictionary
        """

        self.network_objects = []
        self.networks = []
        self.env = environment

        # parameters for visualization
        self.node_size = 2200
        self.arc_rad = 0.1

        self.from_to = {
            (e_def.from_level, tl): e_def.rewards
            for e_def in self.env.edges
            for tl in e_def.to_levels
        }
        self.start_node = None

    def generate(self, n_networks):
        """
        Using the functions defined above this method generates networks with
        visualization info included.
        The generated network(s) are also saved in a json file at location
        specified by save_path

        """

        # sample and store training networks
        self.networks = []
        for _ in range(n_networks):
            g = self.sample_network()
            net = nx.json_graph.node_link_data(g)

            # NEW: shuffle randomly the order of the nodes in circular layout
            pos = nx.circular_layout(g)
            node_order = list(g.nodes(data=True))
            random.shuffle(node_order)
            random_pos = {}
            for a, b in zip(node_order, [pos[node] for node in g]):
                random_pos[a[0]] = b

            pos_map = {
                k: {"x": v[0] * 100, "y": v[1] * -1 * 100}
                for k, v in random_pos.items()
            }

            # NEW: add vertices for visualization purposes
            for ii, e in enumerate(g.edges()):
                if reversed(e) in g.edges():
                    net["links"][ii]["arc_type"] = "curved"
                    arc = nx.draw_networkx_edges(
                        g,
                        random_pos,
                        edgelist=[e],
                        node_size=self.node_size,
                        connectionstyle=f"arc3, rad = {self.arc_rad}",
                    )
                else:
                    net["links"][ii]["arc_type"] = "straight"
                    arc = nx.draw_networkx_edges(
                        g, random_pos, edgelist=[e], node_size=self.node_size
                    )

                vert = arc[0].get_path().vertices.T[:, :3] * 100

                net["links"][ii]["source_x"] = vert[0, 0]
                net["links"][ii]["source_y"] = -1 * vert[1, 0]
                net["links"][ii]["arc_x"] = vert[0, 1]
                net["links"][ii]["arc_y"] = -1 * vert[1, 1]
                net["links"][ii]["target_x"] = vert[0, 2]
                net["links"][ii]["target_y"] = -1 * vert[1, 2]

            network_id = hashlib.md5(
                json.dumps(net, sort_keys=True).encode("utf-8")
            ).hexdigest()

            c = Counter([e["source"] for e in net["links"]])

            if (
                all(value == self.env.n_edges_per_node for value in c.values())
                and len(list(c.keys())) == self.env.n_nodes
            ):
                create_network = self.create_network_object(
                    pos_map=pos_map,
                    n_steps=self.env.n_steps,
                    network_id=network_id,
                    **net,
                )
                self.networks.append(create_network)
                print(f"Network {len(self.networks)} created")
            else:
                print(
                    f"counter {c}, nodes are {list(c.keys())} "
                    f"(n={len(list(c.keys()))})"
                )

        self.network_objects = [Network(**n) for n in self.networks]
        return self.networks

    # individual network building functions
    #######################################
    def add_link(self, G, from_node, to_node):
        from_level = G.nodes[from_node]["level"]
        to_level = G.nodes[to_node]["level"]
        reward_idx = random.choice(self.from_to[(from_level, to_level)])
        reward = self.env.rewards[reward_idx]
        G.add_edge(
            from_node,
            to_node,
            reward=reward.reward,
            reward_idx=reward_idx,
            color=reward.color,
        )

    @staticmethod
    def add_new_node(G, level):
        idx = len(G)
        name = string.ascii_uppercase[idx % len(string.ascii_lowercase)]
        G.add_node(idx, name=name, level=level)
        return idx

    @staticmethod
    def nodes_random_sorted_by_in_degree(G, nodes):
        return sorted(
            nodes, key=lambda n: G.in_degree(n) + random.random() * 0.1, reverse=False
        )

    @staticmethod
    def nodes_random_sorted_by_out_degree(G, nodes):
        return sorted(
            nodes,
            key=lambda n: G.nodes[n]["level"]
            + G.out_degree(n) * 0.1
            + random.random() * 0.01,
            reverse=False,
        )

    def edge_is_allowed(self, G, from_node, to_node):
        if from_node == to_node:
            return False
        if to_node in G[from_node]:
            return False
        from_level = G.nodes[from_node]["level"]
        to_level = G.nodes[to_node]["level"]
        return (from_level, to_level) in self.from_to

    def allowed_nodes(self, G, nodes, from_node):
        return [node for node in nodes if self.edge_is_allowed(G, from_node, node)]

    def assign_levels(self, graph):
        levels = self.env.levels.copy()
        # min number of nodes to each level
        for level in levels:
            level.n_nodes = level.min_n_nodes

        # total number of nodes per levels
        n_nodes = sum([level.n_nodes for level in levels])
        assert n_nodes <= self.env.n_nodes

        # spread missing nodes over levels
        for i in range(self.env.n_nodes - n_nodes):
            # possible levels
            possible_levels = [
                level
                for level in levels
                if (level.max_n_nodes is None) or (level.n_nodes < level.max_n_nodes)
            ]
            # choose level
            level = random.choice(possible_levels)
            # add node to level
            level.n_nodes += 1

        # add nodes to graph
        for level in levels:
            for _ in range(level.n_nodes):
                node_idx = self.add_new_node(graph, level.idx)
                if level.is_start and self.start_node is None:
                    self.start_node = node_idx

    def sample_network(self):
        graph = nx.DiGraph()

        self.assign_levels(graph)
        for i in range(self.env.n_edges_per_node * self.env.n_nodes):
            allowed_from_nodes = [
                n
                for n in graph.nodes
                if not graph.out_degree(n) >= self.env.n_edges_per_node
            ]
            if len(allowed_from_nodes) == 0:
                raise ValueError("No allowed nodes to connect from.")
            from_node = self.nodes_random_sorted_by_out_degree(
                graph, allowed_from_nodes
            )[0]
            allowed_to_nodes = self.allowed_nodes(graph, graph.nodes, from_node)
            if len(allowed_to_nodes) == 0:
                raise ValueError("No allowed nodes to connect to.")
            to_node = self.nodes_random_sorted_by_in_degree(graph, allowed_to_nodes)[0]
            self.add_link(graph, from_node, to_node)
        return graph

    @staticmethod
    def parse_node(name, pos_map, id, **kwargs):
        return {
            "node_num": id,
            "display_name": name,
            "node_size": 3,
            "level": kwargs["level"],
            **pos_map[id],
        }

    @staticmethod
    def parse_link(
        source,
        target,
        reward,
        reward_idx,
        arc_type,
        source_x,
        source_y,
        arc_x,
        arc_y,
        target_x,
        target_y,
        **_,
    ):
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
            "target_y": target_y,
        }

    def create_base_network_object(
        self, pos_map, starting_node=0, *, nodes, links, network_id, n_steps, **kwargs
    ):
        return {
            "network_id": network_id,
            "nodes": [self.parse_node(pos_map=pos_map, **n) for n in nodes],
            "edges": [self.parse_link(**l) for l in links],
            "starting_node": starting_node,
        }

    def create_network_object(self, **kwargs):
        network = self.create_base_network_object(**kwargs)
        return network

    def save_as_json(self):
        return json.dumps(self.networks)
