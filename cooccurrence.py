import argparse
import csv
import sys
import numpy as np
import pickle


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

def get_anchor_words(anchor_fname):
    try:
        parsed_data = parse_tsv(anchor_fname)
    except FileNotFoundError:
        print(f"The file '{anchor_fname}' was not found.")
        sys.exit()
    
    #maps word key to word_index value
    anchor_words = {}
    #skip first row because it is labels
    for row in parsed_data[1:]:
        index = int(row[0])
        word = row[2]
        anchor_words[word] = index
    
    return anchor_words


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

def count_post(post, anchors):
    user = post[0]
    text = post[1]
    counts = [0] * len(anchors)
    #for each anchor, count how often it appears in the text
    for anchor, anchor_idx in anchors.items():
        text_list = text.split()
        count = text_list.count(anchor)
        counts[anchor_idx] = count
    return Post(user, text, counts)


def count_posts(posts, anchors):
    rtn = []
    for post in posts:
        rtn.append(count_post(post, anchors))

    return rtn

def counts_to_matrices(posts):
    #list of tuples (user id, l x l matrices where l = len(anchors))
    matrices = []
    for post in posts:
        num_anchors = len(post.counts)
        matrix = np.zeros((num_anchors, num_anchors))
        for i in range(num_anchors):
            for j in range(num_anchors):
                if(i == j):
                    #no point in crossing a word with itself here
                    continue
                matrix[i, j] = post.counts[i] * post.counts[j]
        matrices.append((post.user, matrix))
    return matrices

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


#currently has slight inaccuracy at 10^-15 sometimes
#takes a matrix and target sensitivity 
#note that full matrix is scaled by 2*target since the matrix double counts each edge
def scale_matrix(matrix, target):
    #is this while loop necessary? Seems like rounding errors mean some matrices sum
    # to very slightly higher with just one scaling, but the sums drop with a loop
    
    #this approach causes everything to be under 20, but with some being significantly lower
    # while(matrix.sum() > 2*target): 
    #     total_weight = matrix.sum()
    #     if total_weight == 0:
    #         #nothing to scale, avoid dividing by 0
    #         return matrix
    #     #the matrix double counts each value, so we scale by 2*target sensitivity
    #     scaling_factor = (2*target) / total_weight
    #     matrix = matrix * scaling_factor
    # return matrix

    #this approach
    total_weight = matrix.sum()
    if total_weight == 0:
        #nothing to scale, avoid dividing by 0
        return matrix
    #the matrix double counts each value, so we scale by 2*target sensitivity
    scaling_factor = (2*target) / total_weight
    matrix = matrix * scaling_factor
    return matrix

def sum_user_matrices(matrices):
    matrix_list = list(matrices.values())
    matrix0 = matrix_list[0]
    
    complete_matrix = np.zeros(matrix0.shape)
    for m in matrix_list:
        complete_matrix += m
    return complete_matrix

def add_noise(matrix, scale):
    #loc = 0, scale = scale, size = matrix.shape
    rng = np.random.default_rng()
    noise = rng.laplace(loc = 0.0, scale = scale, size = matrix.shape)
    new_matrix = matrix + noise
    return new_matrix



def main():
    parser = argparse.ArgumentParser(description='small script to create co-occurence matrices per user given proper datasets')
    parser.add_argument('anchor', type=str, help='The file name the anchor words')
    parser.add_argument('posts', type=str, help="The file name of the posts")
    parser.add_argument('--out', type=str, help="The prefix of the output file")
    parser.add_argument('--in_matrix', type=str, help="Filename of serialized user matrices, to avoid recalculation")
    
    args = parser.parse_args()

    if args.in_matrix:
        anchor_words = get_anchor_words(args.anchor)
        print("Got anchor words")
        
        with open(args.in_matrix, 'rb') as f:
            user_matrices = pickle.load(f)
        print("Loaded user matrices from file")
        # print(len(user_matrices.keys()))

        # reverse_anchor_words = [""] * len(anchor_words)
        # for anchor, anchor_idx in anchor_words.items():
        #     reverse_anchor_words[anchor_idx] = anchor

        # key = "da0a4d57dd0fc1e5c5ab97dc0d0ef7b079c8fcbf64d40495e1a0c5e227119476"
        # test_matrix = user_matrices.get(key)
        # nonzero_indices = np.nonzero(test_matrix)
        # row_indices = nonzero_indices[0]
        # col_indices = nonzero_indices[1]
        # for r, c in zip(row_indices, col_indices):
        #     print(f"{reverse_anchor_words[r]}, {reverse_anchor_words[c]}")

    else:

        #key anchor word, value index
        anchor_words = get_anchor_words(args.anchor)
        print("Got anchor words")
        # reverse_anchor_words = [""] * len(anchor_words)
        # for anchor, anchor_idx in anchor_words.items():
        #     reverse_anchor_words[anchor_idx] = anchor

        #key user hash, value post text
        uncounted_posts = get_posts(args.posts)
        print("Got post data")

        #returns list of Post objects
        counted_posts = count_posts(uncounted_posts, anchor_words)
        print("Counted anchor words in posts")
        # print(counted_posts[10])
        # for i, count in enumerate(counted_posts[10].counts):
        #     if(count > 0):
        #         print(reverse_anchor_words[i])

        #each item in this list is (user hash, co-occurrence matrix for the post)
        post_matrices = counts_to_matrices(counted_posts)
        print("Converted counts into per-post co-occurrence matrices")
        # print(post_matrices[0])
        # nonzero_indices = np.nonzero(post_matrices[0][1])
        # row_indices = nonzero_indices[0]
        # col_indices = nonzero_indices[1]
        # print(counted_posts[0].text)
        # for r, c in zip(row_indices, col_indices):
        #     print(f"{reverse_anchor_words[r]}, {reverse_anchor_words[c]}")

        #hashmap where key= user hash, value = post matrix
        user_matrices = group_post_matrices(post_matrices)
        print("Determined per-user co-occurrence matrices")
        # print(len(user_matrices.keys()))
        # key = "da0a4d57dd0fc1e5c5ab97dc0d0ef7b079c8fcbf64d40495e1a0c5e227119476"
        # test_matrix = user_matrices.get(key)
        # nonzero_indices = np.nonzero(test_matrix)
        # row_indices = nonzero_indices[0]
        # col_indices = nonzero_indices[1]
        # for r, c in zip(row_indices, col_indices):
        #     print(f"{reverse_anchor_words[r]}, {reverse_anchor_words[c]}")

        if args.out:
            serialize_user_matrices(user_matrices, args.out)
            print("Serialized matrix")

    #user matrices are either calculated or loaded in from file by this point

    #scale matrices to sensitivity 10 (total sum 20)
    for user, matrix in user_matrices.items():
        new_matrix = scale_matrix(matrix, 10)
        user_matrices[user] = new_matrix
        # sum = new_matrix.sum()
        # if sum > 20.0:
        #     print(sum)
    print("Scaled user matrices to sensitivity 10")

    complete_matrix = sum_user_matrices(user_matrices)
    print("Combined user matrices")

    #for now, sensitivity = 10, epsilon = 5, so scale = 2
    noisy_matrix = add_noise(complete_matrix,2)
    print("Added noise to combined matrix")
    # print(noisy_matrix)

    

    






if __name__ == "__main__":
    main()
