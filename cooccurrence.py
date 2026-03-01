import argparse
import csv
import sys

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

def main():
    parser = argparse.ArgumentParser(description='small script to create co-occurence matrices per user given proper datasets')
    parser.add_argument('anchor', type=str, help='The file name the anchor words')
    parser.add_argument('posts', type=str, help="The file name of the posts")
    parser.add_argument('--out', type=str, help="The prefix of the output file")
    
    args = parser.parse_args()

    #key anchor word, value index
    anchor_words = get_anchor_words(args.anchor)
    # reverse_anchor_words = [""] * len(anchor_words)
    # for anchor, anchor_idx in anchor_words.items():
    #     reverse_anchor_words[anchor_idx] = anchor

    #key user hash, value post text
    uncounted_posts = get_posts(args.posts)

    #returns list of Post objects
    counted_posts = count_posts(uncounted_posts, anchor_words)
    # print(counted_posts[10])
    # for i, count in enumerate(counted_posts[10].counts):
    #     if(count > 0):
    #         print(reverse_anchor_words[i])








if __name__ == "__main__":
    main()
