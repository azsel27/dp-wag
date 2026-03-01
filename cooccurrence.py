import argparse
import csv
import sys

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

    posts = {}
    for row in parsed_data:
        user_hash = row[0]
        post_content = row[1]
        posts[user_hash] = post_content
    
    return posts

def main():
    parser = argparse.ArgumentParser(description='small script to create co-occurence matrices per user given proper datasets')
    parser.add_argument('anchor', type=str, help='The file name the anchor words')
    parser.add_argument('posts', type=str, help="The file name of the posts")
    parser.add_argument('--out', type=str, help="The prefix of the output file")
    
    args = parser.parse_args()

    #key anchor word, value index
    anchor_words = get_anchor_words(args.anchor)

    #key user hash, value post text
    posts = get_posts(args.posts)

    # print(f"Anchor: {args.anchor}")
    # print(f"Posts: {args.posts}")
    # print(f"Out: {args.out}")




if __name__ == "__main__":
    main()
