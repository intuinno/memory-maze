import numpy as np
from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt

import igraph
from matplotlib import cm, colors
import os
import pickle


custom_colors = (
    np.array(
        [
            [214, 214, 214],
            [85, 35, 157],
            [253, 252, 144],
            [114, 245, 144],
            [151, 38, 20],
            [239, 142, 192],
            [214, 134, 48],
            [140, 194, 250],
            [72, 160, 162],
        ]
    )
    / 256
)
if not os.path.exists("figures"):
    os.makedirs("figures")
    
    

def plot_graph(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=10, ground_truth_positions=None
):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)
    


    # Calculate the ground truth positions        
    # Initialize layout
    if ground_truth_positions is not None:
        # Ensure positions match the number of vertices
        assert len(ground_truth_positions) == len(states), "Mismatch between vertices and ground truth positions"
        # Initialize a dictionary to store the sum of positions and count for each state
        state_positions = {state: {'sum': np.zeros(2), 'count': 0} for state in states}

        # Iterate through the states and accumulate the positions
        for state, position in zip(states, ground_truth_positions):
            state_positions[state]['sum'] += position
            state_positions[state]['count'] += 1

        # Calculate the average position for each state
        average_positions = {state: pos['sum'] / pos['count'] if pos['count'] > 0 else np.zeros(2)
                            for state, pos in state_positions.items()}


        g = igraph.Graph.Adjacency((A > 0).tolist())
        node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]
        if multiple_episodes:
            node_labels -= 1
        colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
        intiial_positions = [average_positions[state] * 10000 for state in v]
        layout = igraph.Layout(intiial_positions)
        layout = g.layout_kamada_kawai()
        
        # g.vs["x"] = [pos[0] for pos in layout]
        # g.vs["y"] = [pos[1]  for pos in layout]
        # Update layout using Kamada-Kawai starting from ground truth
        # Convert vertex attributes into initial layout coordinates
        # layout = g.layout_kamada_kawai()

    else:
        # Default to Kamada-Kawai layout
        layout = g.layout("kamada_kawai")

    # Plot the graph
    out = igraph.plot(
        g,
        output_file,
        layout=layout,
        vertex_color=colors,
        # vertex_label=v,
        vertex_size=vertex_size,
        margin=0,
    )

    return out


def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x * 0, store_messages=True
    )
    return mess_fwd


# Load the saved environment data
data = np.load("data/small_env_5_5_3actions_100k_low_orient.npz")
# Load the cluster indices
cluster_indices = np.load("data/cluster_indices_10k.npy")


n_emissions = cluster_indices.max() + 1

# Calculate unique values and their counts
unique_values, counts = np.unique(cluster_indices, return_counts=True)

n_clones = np.ones(n_emissions, dtype=np.int64) * 70

x = cluster_indices.astype(np.int64)
a = data['actions'].astype(np.int64) - 1


with open('data/chmm_100k.pkl', 'rb') as file:
    chmm = pickle.load(file)

graph = plot_graph(
    chmm, x, a, output_file="figures/memory_maze_graph.pdf", cmap=cm.Reds, ground_truth_positions=data['agent_pos']
)
graph