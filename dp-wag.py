import argparse
import csv
import sys
import numpy as np
import pickle
import networkx as nx 
import igraph as ig
import leidenalg as la
from collections import Counter
import matplotlib.pyplot as plt
from pyvis.network import Network 
import matplotlib.colors as mcolors 
import configparser
import string
import json
from pathlib import Path

import nltk
from nltk.corpus import stopwords


# This script is intended to be a proof-of-concept for differentially private
# word association graph generation and community detection. 

class Post:
    def __init__(self, user, text, counts):
        self.user = user
        self.text = text
        self.counts = counts

    def __str__(self):
        return f"User {self.user}, text: {self.text}, counts: {self.counts}"

def parse_tsv(filename):
    data = []
    # Open the file and specify the tab delimiter
    with open(filename, newline='', encoding='utf-8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def remove_stop_words(anchor_words_dict):
    #convert anchors to set
    anchor_list = set(anchor_words_dict)

    #download stop words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    #list comprehension to filter
    filtered_anchors = [word for word in anchor_list if word not in stop_words]
    return filtered_anchors


def get_anchor_words(anchor_fname):
    try:
        parsed_data = parse_tsv(anchor_fname)
    except FileNotFoundError:
        print(f"The file '{anchor_fname}' was not found.")
        sys.exit()
    
    #maps word key to word_index value
    anchor_words = {}
    index = 0
    #skip first row because it is labels
    for row in parsed_data[1:]:
        word = row[2]
        anchor_words[word] = index
        index += 1

    filtered_anchors = remove_stop_words(anchor_words)
    filtered_anchor_words = {filtered_anchors[idx]: idx for idx in range(len(filtered_anchors))}

    print(f"NUM ANCHORS TOTAL: {len(anchor_words)}")
    print(f"NUM FILTERED ANCHORS TOTAL: {len(filtered_anchor_words)}")
    
    return filtered_anchor_words

#parse posts into list of tuples [(user hash, post text), ...]
def get_posts(post_fname):
    try:
        parsed_data = parse_tsv(post_fname)
    except FileNotFoundError:
        print(f"The file '{post_fname}' was not found.")
        sys.exit()

    posts = []
    for row in parsed_data:
        user_hash = row[0]
        post_content = row[1]
        posts.append((user_hash, post_content))
    
    return posts


def full_post_to_matrix(post, anchors):
    post_user = post[0]
    post_text = post[1]
    post_words = [word.lower().strip(string.punctuation) for word in post_text.split()]
    num_anchors = len(anchors)

    anchor_counts = [0] * num_anchors
    for word in post_words:
        anchor_idx = anchors.get(word)
        if anchor_idx == None:
            #not an anchor
            continue
        anchor_counts[anchor_idx] += 1
    
    matrix = np.zeros((num_anchors, num_anchors))
    for i in range(num_anchors):
        for j in range(num_anchors):
            if(i == j):
                #no point in crossing a word with itself here
                continue
            matrix[i, j] = anchor_counts[i] * anchor_counts[j]
    return (post_user, matrix)

def posts_to_matrices_full_post(posts, anchors):
    matrices = []
    for post in posts:
        post_user, matrix = full_post_to_matrix(post, anchors)
        matrices.append((post_user, matrix))
    return matrices

def posts_to_matrices_param(posts, anchors, window_size):
    if window_size == -1:
        return posts_to_matrices_full_post(posts, anchors)
    matrices = []
    for post in posts:
        post_user = post[0]
        post_text = post[1]
        post_words = [word.lower().strip(string.punctuation) for word in post_text.split()]
        num_words = len(post_words)
        num_anchors = len(anchors)
        matrix = np.zeros((num_anchors, num_anchors))

        #iterate through window    
        for i in range(num_words - 1):
            word = post_words[i]
            idx = anchors.get(word)
            if idx is None:
                #not an anchor
                continue
            #iterate through full window or until end of the post
            for j in range(i+1, min(i+window_size+1, num_words)): 
                next_word = post_words[j]
                next_idx = anchors.get(next_word)
                if next_idx is None or next_idx == idx:
                    #not an anchor or same word
                    continue
                #else both anchors, increment the matrix
                matrix[idx, next_idx] += 1
                matrix[next_idx, idx] += 1 #symmetrical for completeness
            matrices.append((post_user, matrix))
    return matrices


# consolidate various post matrices into per-user matrices
def group_post_matrices(post_matrices):
    user_matrices = {}
    for post_tuple in post_matrices:
        user_hash = post_tuple[0]
        post_matrix = post_tuple[1]
        #get or create matrix m based on user hash
        user_matrix = user_matrices.get(user_hash, np.zeros(post_matrix.shape))
        #add post matrix to m
        user_matrix += post_matrix
        #assign user_matrices[user_matrix] = post_matrix + m
        user_matrices[user_hash] = user_matrix
    return user_matrices

def serialize_user_matrices(data, fname):
    if not fname:
        fname = "matrix_out.pickle"
    
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# currently has slight inaccuracy on order of 10^-15 sometimes
# note that full matrix is scaled by 2*target since the matrix double counts each edge
# scales a matrix to a target value so that 
# total sum of the matrix as an adjacency graph <= target
def scale_matrix(matrix, target):
    total_weight = matrix.sum()
    if total_weight == 0:
        #nothing to scale, avoid dividing by 0
        return matrix
    if total_weight < target:
        return matrix

    scaling_factor = (2*target) / total_weight
    matrix = matrix * scaling_factor
    return matrix

# combine user matrices into single unified matrix
def sum_user_matrices(matrices):
    matrix_list = list(matrices.values())
    matrix0 = matrix_list[0]
    
    complete_matrix = np.zeros(matrix0.shape)
    for m in matrix_list:
        complete_matrix += m
    return complete_matrix

# input : {"anchor word": index, ...}
# output : [anchor0, anchor1, ...]
def get_reverse_anchor_words(anchor_words):
    reverse_anchor_words = [""] * len(anchor_words)
    for anchor, anchor_idx in anchor_words.items():
        reverse_anchor_words[anchor_idx] = anchor
    return reverse_anchor_words

# add Laplace noise to the values of a matrix
def add_noise(matrix, scale):
    rng = np.random.default_rng()
    noise = rng.laplace(loc = 0.0, scale = scale, size = matrix.shape)
    new_matrix = matrix + noise
    return new_matrix

# extract top k edges from the matrix, associate them with the corresponding anchor
def get_top_k(matrix, k, anchor_words):
    reverse_anchors = get_reverse_anchor_words(anchor_words)

    #k= -1 to exclude the diagonal, 0 if noisy diagonal should be included
    lt_rows, lt_cols = np.tril_indices_from(matrix, k=-1)

    #values from the lower triangle of the matrix as a list
    lt_matrix_values = matrix[lt_rows, lt_cols]

    #unsorted list of top k indices
    top_k_indices = np.argpartition(lt_matrix_values, -k)[-k:]
    #sort and reverse top k indices
    sorted_indices = np.argsort(lt_matrix_values[top_k_indices])[::-1]
    top_k_indices = top_k_indices[sorted_indices]
    
    #tuples of matrix coordinates with the values 
    index_values = [(lt_rows[i], lt_cols[i], lt_matrix_values[i]) for i in top_k_indices]
    #tuples of anchor words with the weight of their edge
    word_values = [(reverse_anchors[i[0]], reverse_anchors[i[1]], i[2]) for i in index_values]

    return word_values
    
# serialize top k edges 
def write_output(top_k, fname):
    with open(fname, 'w', newline='') as out:
        writer = csv.writer(out)
        headers = ['Word1', 'Word2', 'Edge weight']
        writer.writerow(headers)
        writer.writerows(top_k)


# extract communities from the private matrix, output is list of lists of indices,
# where communities are the sublists
def communities_from_noisy_matrix_leidenalg(m, resolution = 1.0, threshold = 0.0):
    #to match the get_top_k weights, we want to examine the lower half of the matrix (aka the upper half of the transposition)
    working_matrix = np.array(m, dtype=np.float64, copy=True)


    # 0 out diagonal and any edges below the threshold
    np.fill_diagonal(working_matrix, 0)
    working_matrix = np.where(working_matrix < threshold, 0, working_matrix)

    g = ig.Graph.Weighted_Adjacency(working_matrix.T, mode="upper", attr="weight")
    g.es.select(weight=0).delete()

    if resolution == 1.0: #should this be resolution?
        partition = la.find_partition(g,
                                      la.ModularityVertexPartition,
                                      weights='weight',
                                      n_iterations=-1,
                                      )
    else:
        partition = la.find_partition(g,
                                      la.RBConfigurationVertexPartition,
                                      weights='weight',
                                      resolution_parameter=resolution,
                                      n_iterations=-1,
                                      )

    print(f"Partition quality: {partition.quality()}")
    communities = [list(comm) for comm in partition]
    return communities


def lowest_weight_edge(g):
    return min(g.edges(data="weight"), key=lambda e: e[2])[:2]

def most_central_edge(g):
    #Invert edges bc NX treats them as distance not similarity
    inverted_weights = {
        (u,v): 1.0/w if w > 0 else float("inf")
        for u, v, w in g.edges(data="weight")
    }
    centrality = nx.algorithms.centrality.edge_betweenness_centrality(g, weight="weight")
    return max(centrality, key=centrality.get)

# networkx implementation, currently using Girvan-Newman
def communities_from_noisy_matrix_networkx(m, threshold = 0.0):
    #to match the get_top_k weights, we want to examine the lower half of the matrix (aka the upper half of the transposition)
    working_matrix = np.array(m, dtype=np.float64, copy=True)


    # 0 out diagonal and any edges below the threshold
    np.fill_diagonal(working_matrix, 0)
    working_matrix = np.where(working_matrix < threshold, 0, working_matrix)

    n = working_matrix.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))

    #lower triangle, no diagonal
    rows, cols = np.tril_indices(n, k=-1)
    weights = working_matrix[rows, cols]
    mask = weights > 0
    g.add_weighted_edges_from(
        zip(rows[mask].tolist(), cols[mask].tolist(), weights[mask].tolist())
    )

    if g.number_of_edges == 0:
        # thresholding too strong for any edges
        return [[node] for node in g.nodes()]
    
    partition_generator = nx.community.girvan_newman(g, most_valuable_edge=lowest_weight_edge)
    # partition_generator = nx.community.girvan_newman(g, most_valuable_edge=most_central_edge)

    best_partition = None
    #TODO for now, modularity. should try performance and coverage
    best_quality = float("-inf")

    limit = n
    for i, partition in enumerate(partition_generator):
        communities = [list(c) for c in partition]
        # q_performance, q_coverage = partition_quality(g, communities) #returns a 2-tuple
        q = nx.algorithms.community.modularity(g, communities, weight = "weight")
        if q > best_quality:
            best_quality = q
            best_partition = communities
        # if q_performance > best_quality:
        #     best_quality = q_performance
        #     best_partition = communities
        if len(communities) >= limit:
            break

    print(f"Partition quality: {best_quality}")
    return best_partition

# transform list of community sublists into same format but strings instead
# of indices
def match_anchors_to_communities(anchors, comm):
    rev_anchors = get_reverse_anchor_words(anchors)
    return [[rev_anchors[i] for i in c] for c in comm]

# serialize communities
def write_out_partitions(comms, fname):
    with open(fname, 'w', newline='') as out:
        writer = csv.writer(out)
        headers = ['Word', 'Partition index']
        writer.writerow(headers)
        for i, comm in enumerate(comms):
            for word in comm:
                writer.writerow((word, i))

def format_partition(time, community_name, partition, anchor_dict, m):
    # [date range] [community] [story hook] [normalized edge weight]

    story_hook = ",".join(partition)
    
    #get list of indices from anchor_dict
    indices = [anchor_dict[key] for key in partition]

    # r > c keeps us in the lower triangle of the matrix
    pairs = [(r,c) for r in indices for c in indices if r > c]
    edges = []
    for pair in pairs:
        edges.append(m[pair[0], pair[1]])
    avg_edge_weight = np.round(np.mean(edges), 3)
        

    #join comm_name, story_hook, avg_edge_weight and combine
    output_line = [time, community_name, story_hook, str(avg_edge_weight)]
    return output_line


def write_out_detailed_partitions(partitions, anchors, community_name, time, matrix,fname):
    file_path = Path(fname)
    file_exists = file_path.is_file()
   

    #TODO this is not a great way to combine the data across runs
    with open(fname, 'a', newline='') as out:
        writer = csv.writer(out, delimiter='\t')
        file_path = Path(fname)
        if not file_exists:
            headers = ['Time','Community', 'Story hook', 'Avg edge weight']
            writer.writerow(headers)
        for i, part in enumerate(partitions):
            if len(part) < 2:
                continue
            partition_line = format_partition(time, community_name, part, anchors, matrix) 
            writer.writerow(partition_line)

# ----------------------- Distribution code ---------------
def get_distribution_buckets_from_matrix(matrix):
    #ignore edges from word to itself again
    lt_rows, lt_cols = np.tril_indices_from(matrix, k=-1)
    subset = matrix[lt_rows, lt_cols]
    rounded_subset = np.round(subset, decimals=0)
    counts = Counter(rounded_subset)
    return counts

def get_bar_graph_from_counts(counts):
    x_vals = list(counts.keys())
    y_vals = list(counts.values())
    plt.bar(x_vals, y_vals, width=1, align='center')
    plt.xlabel('Weights')
    plt.ylabel('Counts')
    # plt.xticks(x_vals)
    plt.xlim(-20,80) #this could be done procedurally but for now, hardcoded
    plt.show()


# ----------------------Visualization code-------------------
#This section of code is not up to date or in active use

# counts the number of times a node crosses into other communities  
def count_comm_crossings(comm_list, adj_matrix):
    node_to_comm = {} #lookup from node index to community index
    for k, members in enumerate(comm_list):
        for node in members:
            node_to_comm[node] = k

    n = adj_matrix.shape[0]
    crossings = [0] * n
    for i in range(n):
        self_comm = node_to_comm.get(i)
        foreign = {node_to_comm[j] for j in range(i) if adj_matrix[i, j] != 0 and node_to_comm.get(j) != self_comm}
        crossings[i] = len(foreign)
    return crossings

# generate a visualization of the communities. this halts execution of the script until
# the visualization is closed. This code is pretty rough for now and being experimented
# with to generate better results for different situations. Unfortunately, 
# a lot of the parameters are hardcoded rather than derived from properties of the
# matrix or communities right now. 
def generate_visualization_partitions(comm, anchor_list, adj_matrix, threshold = 0.0):
    G = nx.Graph()
    n = len(adj_matrix)

    rev_anchor = get_reverse_anchor_words(anchor_list)
    for i, word in enumerate(rev_anchor):
        G.add_node(i, label=word)
    
    #scale edges by community crossover
    crossings = count_comm_crossings(comm, adj_matrix)
    denominators = [c if c > 0 else 1 for c in crossings] #don't divide by 0
    for i in range(n):
        for j in range(i):
            if adj_matrix[i, j] > threshold:
                target_weight = adj_matrix[i, j] / (denominators[i] * denominators[j])
                G.add_edge(i, j, weight=target_weight)

  


    #draft 1: i_w 15.0, k=15, iter=300
    #draft 2: 20.0, 20, 400
    #draft 3: 50.0, 15, 400
    #draft 4, 55.0, 12, 400
    #dr 5: 60.0, 15, 400
    #dr 6: 60.0, 20, 400, well grouped except for 2 partitions (seed 42 for all)
    #30 for adjacency 1
    intra_weight = 30.0 #used 5.0 for bluesky, 10.0 for reddit. clusters a community together
    for community in comm:
        for i in range(len(community)):
            for j in range(i+1, len(community)):
                if G.has_edge(community[i], community[j]):
                    # Boost existing edge weight
                    G[community[i]][community[j]]['weight'] += intra_weight
                else:
                    # Add virtual edge
                    G.add_edge(community[i], community[j], weight=intra_weight, seed=42)
    
    node_colors = ['lightgray'] * n
    community_colors = list(mcolors.TABLEAU_COLORS.values())

    #increase k to make communities spread out more. used k=2 for bluesky, k=15 for reddit
    pos = nx.spring_layout(G, k=20, iterations=400) #TODO fix seed?

    for comm_idx, community in enumerate(comm):
        color = community_colors[comm_idx % len(community_colors)]
        for node in community:
            node_colors[node] = color
    

    fig, ax = plt.subplots(figsize=(12, 8))
    real_edges = [(i, j) for i in range(n) for j in range(i) 
                  if adj_matrix[i, j] > threshold]
    nx.draw_networkx_edges(G, pos, edgelist=real_edges, 
                          alpha=0.2, width=0.5, edge_color='gray', ax=ax)
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, 
                          edgecolors='white', linewidths=1.5, ax=ax)
    
    labels = {i: rev_anchor[i] for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight='bold', ax=ax)

    legend_elements = []
    for i, community in enumerate(comm):
        color = community_colors[i % len(community_colors)]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                              facecolor=color, edgecolor='white',
                              label=f'Community {i+1} ({len(community)} nodes)'))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.axis('off')
    plt.tight_layout()
    plt.show()





def main():
    #read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    #get anchor words from input file
    anchor_input_fname = config['input']['anchor_input']
    anchor_words = get_anchor_words(anchor_input_fname)
    print(f"Read input anchor words")

    #read posts from input file
    posts_input_fname = config['input']['post_input']
    uncounted_posts = get_posts(posts_input_fname)
    print("Read input posts")

    #convert posts to co-occ matrices
    window_size = int(config['statistics']['cooccurrence_window'])

    #window_size = -1 -> full post co-occurrence
    post_matrices = posts_to_matrices_param(uncounted_posts, anchor_words, window_size)
    print("Generated adjacency co-occurence matrices for posts")
    
    #group post-level matrices
    user_matrices = group_post_matrices(post_matrices)
    print("Determined per-user co-occurrence matrices")

    #scale matrices
    target_sens = float(config['laplace']['target_sensitivity'])
    for user, matrix in user_matrices.items():
        new_matrix = scale_matrix(matrix, target_sens)
        user_matrices[user] = new_matrix
    print("Scaled user matrices to sensitivity "+str(target_sens))

    #aggregate matrices
    complete_matrix = sum_user_matrices(user_matrices)
    print("Aggregated user matrices")

    #add noise
    epsilon = float(config['laplace']['epsilon'])
    scale_factor = target_sens/epsilon
    print("eps is "+str(epsilon))
    print("scale factor is "+str(scale_factor))
    noisy_matrix = add_noise(complete_matrix, scale_factor)
    print("Added noise to matrix")

    #get top k
    k = int(config['statistics']['top_k'])
    top_edges = get_top_k(noisy_matrix, k, anchor_words)
    print("Got top "+ str(k) +" edges")

    #determine communities
    threshold = float(config['leiden']['threshold'])
    resolution = float(config['leiden']['resolution'])

    comm = communities_from_noisy_matrix_networkx(noisy_matrix, threshold = threshold)
    print("Determined partitions")

    #match anchors to communities
    comm_anchors = match_anchors_to_communities(anchor_words, comm)
    print("Matched words to partitions")

    #write top k
    top_k_fname = config['output']['top_k_output']
    write_output(top_edges, top_k_fname)
    print(f"Wrote to file {top_k_fname}")

    #write communities
    comm_fname = config['output']['partition_out']
    write_out_partitions(comm_anchors, comm_fname)
    print(f"Wrote partitions to file {comm_fname}")

    # partitions, anchors, community_name, matrix,fname
    community_name = config['output']['community_name']
    shared_partition_fname = config['output']['shared_partition_fname']
    time_lookup_fname = config['input']['dataset_times']
    with open(time_lookup_fname, 'r') as f:
        time_lookup = json.load(f)
    time = time_lookup.get(community_name, None)
    if time is None:
        print("COULDN'T FIND COMMUNITY IN TIME LOOKUP")
        return
    write_out_detailed_partitions(comm_anchors, anchor_words, community_name, time, noisy_matrix, shared_partition_fname)
    print(f"Added detailed partitions to shared file {shared_partition_fname}")


    

    



if __name__ == "__main__":
    main()
