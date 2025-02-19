from responses import stop
from ..headers import CodingProblem, CodingProblemAndSolution, Solver, AbstractLogged

from typing import Optional, Hashable, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math

@dataclass
class Node(AbstractLogged):
    '''
    Node in the search tree
    '''
    solution: Optional[CodingProblemAndSolution]
    parent:  Optional['Node'] 
    score: float 
    depth: int 
    visits: int 
    action_to_node: dict[Hashable, 'Node'] 
    actions: set 
    value: float 

    def __init__(self, solution: Optional[CodingProblemAndSolution], parent: Optional['Node'] = None, score: float = 0.0, depth: int = 0, value: float = 0.0):
        self.solution = solution
        self.parent = parent
        self.score = score
        self.depth = depth
        self.value = value
        self.action_to_node = {}
        # assign default values of 1.0 to all actions
        self.action_to_logp = defaultdict(lambda: 1.0)
        self.actions = set()
        self.visits = 1
        super().__init__()

    def __str__(self):
        return f"Node(strategy={self.solution}, score={self.score}, depth={self.depth})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.solution)

    def add_child_solution(self, action: Hashable, solution: CodingProblemAndSolution, score: float, logp: float = 1.0) -> 'Node':
        self.logger.debug(f"Adding child solution: {hash(solution)} with score: {score}, action: {action}, logp: {logp} to parent: {hash(self)}")
        child = Node(solution, parent=self, score=score, depth=self.depth+1, value=score)
        self.action_to_node[action] = child
        self.actions.add(action)
        self.action_to_logp [action] = logp
        return child

    def get_best_child_uct(self, is_puct: bool = False) -> Optional['Node']:
        '''
        Returns the best child according to UCT values of the actions. None if no children
        '''
        best_child = None
        best_uct = float('-inf')
        for action, child in self.action_to_node.items():
            if is_puct:
                uct = self.get_puct(action)
            else:
                uct = self.get_uct(action)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        return best_child
    
    def has_unvisited_actions(self) -> bool:
        '''
        Returns True if there are unvisited actions
        '''
        return len(self.actions) > len(self.action_to_node)
    
    def get_random_unvisited_action(self) -> Optional[Hashable]:
        '''
        Returns a random unvisited action. None if no unvisited actions
        '''
        unvisited_actions = self.actions - self.action_to_node.keys()
        if len(unvisited_actions) == 0:
            return None
        return np.random.choice(list(unvisited_actions))
    
    def has_no_children(self) -> bool:
        '''
        Returns True if the node has no children
        '''
        return len(self.action_to_node) == 0

    def has_no_better_child(self) -> bool:
        '''
        Returns whether any child has a better score than the current node
        '''
        for child in self.action_to_node.values():
            if child.get_score() > self.get_score():
                return False
        return True
    
    def get_uct(self, action: Hashable, c: float = math.sqrt(2),) -> float:
        '''
        Computes the UCT value for the given action
        '''
        if action not in self.action_to_node:
            self.logger.debug(f"Action: {action} not in action_to_node")
            return float('inf')
        else:
            child = self.action_to_node[action]
            return child.value + c * np.sqrt(np.log(self.visits) / child.visits)

    def get_puct(self, action: Hashable, c: float = 2.0, cbase: float = 10.0) -> float:
        '''
        Computes the PUCT value for the given action
        '''
        if action not in self.action_to_node:
            self.logger.debug(f"Action: {action} not in action_to_node")
            return float('inf')
        else:
            child = self.action_to_node[action]
            beta = np.log((1 + self.visits + cbase) / cbase) + c
            out = child.value + beta * self.get_pvalue(action) * np.sqrt(np.log(self.visits) / (1+child.visits))
            self.logger.debug(f"PUCT value for action: {action} is {out}")
            return out

    def get_pvalue(self, action: Hashable) -> float:
        '''
        Converts logprob to pvalue. We will need to take exp and normalize it
        '''
        max_logp = max(self.action_to_logp.values())
        print("Action to logp: ", self.action_to_logp)
        out = np.exp(self.action_to_logp[action] - max_logp) / sum(np.exp(logp - max_logp) for logp in self.action_to_logp.values())
        self.logger.debug(f"Pvalue for action: {action} is {out}")
        return out

    def backpropogate(self, alpha: float = 1.0):
        '''
        Backpropogates the value up the tree
        '''
        if self.parent is not None:
            self.parent.value += alpha * max(0, self.value-self.parent.value)
            self.parent.backpropogate()
    
    def get_score(self) -> float:
        '''
        Returns the score of the node
        '''
        return self.score

@dataclass
class Graph(AbstractLogged):
    '''
    Graph of nodes
    '''
    root: Node
    nodes: dict[CodingProblemAndSolution, Node] = field(default_factory=dict)

    def __init__(self, root: Node, is_puct: bool = False):
        self.root = root
        self.nodes = {}
        self.is_puct = is_puct
        super().__init__()

    @staticmethod
    def init_from_seed_solutions(seed_solutions: list[CodingProblemAndSolution]) -> 'Graph':
        '''
        Initializes the graph from seed solutions
        '''
        root = Node(None, score=-1.0)
        graph = Graph(root)
        # create a node for each seed solution and add it to the graph
        for seed_solution in seed_solutions:
            root.add_child_solution(hash(seed_solution), seed_solution, sum(seed_solution.test_results)/len(seed_solution.visible_tests), logp=seed_solution.extra_kwargs['logprob'])
            graph.logger.debug(f"Added seed solution: {hash(seed_solution)} with score: {sum(seed_solution.test_results)/len(seed_solution.visible_tests)}")
            graph.nodes[seed_solution] = root.action_to_node[hash(seed_solution)]
        return graph
    
    def add_child_solution(self, parent_solution: CodingProblemAndSolution, action: Hashable, solution: CodingProblemAndSolution, score: float, logp: float = 1.0) -> Node:
        '''
        Adds a child solution to the parent solution
        '''
        parent_node = self.nodes[parent_solution]
        child_node = parent_node.add_child_solution(action=action, solution=solution, score=score, logp=logp)
        self.nodes[solution] = child_node
        return child_node

    def simulate_trajectory(self, stop_condition: str = "has_unvisited_action") -> Node:
        '''
        Simulates a trajectory from the root node until stop_condition is met

        # stop_condition: Callable[[Node], bool]
        '''
        if stop_condition == "has_unvisited_action":
            def do_stop(node: Node) -> bool:
                return node.has_unvisited_actions()
        elif stop_condition == "has_no_better_child":
            def do_stop(node: Node) -> bool:
                return node.has_no_better_child()
        elif stop_condition == "has_unvisited_actions_or_no_better_child":
            def do_stop(node: Node) -> bool:
                return node.has_unvisited_actions() or node.has_no_better_child()
        else:
            raise ValueError(f"Invalid stop_condition: {stop_condition}")

        cur_node = self.root
        cur_node.visits += 1
        
        self.logger.debug(f"Simulating trajectory")
        # while not cur_node.has_no_children() and not cur_node.has_no_better_child():
        counter = 0
        while not cur_node.has_no_children() and not do_stop(cur_node):
            assert counter < 100, "MCTS Infinite loop trajectory"
            self.logger.debug(f"Current node: {hash(cur_node)}")
            # print out the UCT values for all the children for debugging
            for action, child in cur_node.action_to_node.items():
                self.logger.debug(f"Considering child: {hash(child)}, UCT: {cur_node.get_uct(action)}, visits: {child.visits}, parent visits: {cur_node.visits}, child value: {child.value}")
            cur_node = cur_node.get_best_child_uct(is_puct=self.is_puct)
            if cur_node is None:
                raise ValueError("No children")
            cur_node.visits += 1
            counter += 1
            # print out the current node
            # self.logger.debug(f"Current node: {cur_node}")
        self.logger.debug(f"Simulated trajectory ended at node: {hash(cur_node)}")
        return cur_node
    
    def get_best_score_solution(self) -> Optional[CodingProblemAndSolution]:
        '''
        Returns the best score solution. None if no solutions
        '''
        best_score = float('-inf')
        best_solution = None
        for solution, node in self.nodes.items():
            if node.get_score() > best_score:
                best_score = node.get_score()
                best_solution = solution
        return best_solution

    def get_all_solutions(self) -> list[CodingProblemAndSolution]:
        '''
        Returns all solutions in the graph
        '''
        return list(self.nodes.keys())

    def get_all_best_solutions(self) -> list[CodingProblemAndSolution]:
        '''
        Returns all solutions with the best score
        '''
        best_score = float('-inf')
        best_solutions = []
        for solution, node in self.nodes.items():
            if node.get_score() > best_score:
                best_score = node.get_score()
                best_solutions = [solution]
            elif node.get_score() == best_score:
                best_solutions.append(solution)
        return best_solutions

    def generate_visualization(self):
        '''
        Generates a plot of the graph where:
        - Node color indicates the number of visits.
        - The node's value is displayed on the node.
        '''
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes and edges to the graph
        for node in self.nodes.values():
            G.add_node(node, visits=node.visits, value=node.value)
            for action, child in node.action_to_node.items():
                G.add_edge(node, child)
        
        # Get node attributes for coloring based on visits
        node_visits = nx.get_node_attributes(G, 'visits')
        node_values = nx.get_node_attributes(G, 'value')
        
        # Fallback if some nodes don't have visit counts (optional)
        node_color = [node_visits.get(n, 0) for n in G.nodes]  # Default to 0 if 'visits' is missing
        
        # Define positions for each node using a layout
        pos = nx.spring_layout(G, seed=42)  # Use a layout algorithm for better visualization

        # Color map for node visits
        cmap = plt.cm.viridis
        
        # Create figure and axis
        fig, ax = plt.subplots()

        # Draw the nodes with color based on visits
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=cmap, 
                                    node_size=500, alpha=0.8, ax=ax)
        
        # Add a color bar for node visits
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Number of Visits')

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, ax=ax)
        
        # Add node labels for the value of each node
        node_labels = {n: f"{node_values.get(n, 0):.2f}" for n in G.nodes}  # Default to 0 if 'value' is missing
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', ax=ax)
        
        # Display the plot
        plt.title("Graph Visualization: Node Visits and Values")
        plt.show()




    def generate_interactive_plot(self):
        '''
        Generates and returns a Plotly interactive plot of the graph
        '''
        G = nx.DiGraph()

        node_values = []
        node_texts = []
        edge_texts = []

        for solution, node in self.nodes.items():
            G.add_node(solution, value=node.value, visits=node.visits, label=str(solution))
            node_values.append(node.visits)
            # Display the value of the node as the text inside the node
            node_texts.append(f'{node.value:.2f}')

            for action, child_node in node.action_to_node.items():
                G.add_edge(solution, child_node.solution, action=action)
                edge_texts.append(f'Action: {action}')

        pos = nx.spring_layout(G)

        node_x = [pos[solution][0] for solution in G.nodes()]
        node_y = [pos[solution][1] for solution in G.nodes()]
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_texts
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',  # Show both markers and text
            hoverinfo='none',  # Disable hover info for nodes (or you can customize it)
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_values,
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Node Visits',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2),
            text=node_texts,  # Set the text inside the nodes as the node's value
            textposition='middle center'  # Position the text at the center of the nodes
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Interactive Search Tree Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            annotations=[dict(
                                text="Graph of Nodes and Actions",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002
                            )],
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )
        return fig
