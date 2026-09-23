import configparser
import pandas as pd
import hashlib
import csv

def process_universal_words(df):
    # print(df['Unnamed: 1'])
    universal_words = []
    for row in df['Unnamed: 1']:
        # print(row)
        universal_words.append(row)
    print(universal_words)
    return universal_words

def extract_anchor_words(cell):
    anchor_words = set(cell.split(', '))
    # print(anchor_words)
    return anchor_words

#taken from Claude and modified
def pseudonymize(name: str, salt: str = "", length: int = 64):
    h = hashlib.sha256((salt + name).encode("utf-8")).hexdigest()
    return h[:length]

def extract_posts(df):
    posts = []
    # print(df['Unnamed: 5']) #row 19 of this is screen name
    # print(df['anchor words'])
    #row 19 is start of posts and posters lists, col 5 is poster, col 6 is post
    post_list_slice = df.iloc[19:, 5:7]
    for i, row in post_list_slice.iterrows():
        poster = row.iloc[0]
        pseudonym = pseudonymize(poster)
        # print(f"Row {i}: {poster} -> {pseudonym}")
        post = row.iloc[1]
        # print(f"Row {i}: ({poster}, {post})")
        posts.append((pseudonym, post))
    return posts


def process_misc_tabs(df):
    # print(df)
    #anchor words in cell G2
    # print(df.iloc[0, 6])
    anchor_words = extract_anchor_words(df.iloc[0,6])
    posts = extract_posts(df)
    # print(posts)
    return anchor_words, posts

def output_anchors(anchors, out_fname):
    with open(out_fname, 'w', newline='') as out:
        writer = csv.writer(out, delimiter="\t")
        headers = ['word_index', 'cluster_id', 'word']
        writer.writerow(headers)
        i = 0
        for anchor in anchors:
            #i isn't actually used by dp-wag so just maintaining structure here
            row = [i, i, anchor]
            writer.writerow(row)
            i += 1

def output_posts(posts, out_fname):
    with open(out_fname, 'w', newline='') as out:
        writer = csv.writer(out, delimiter="\t")
        for post_tup in posts:
            row = [post_tup[0], post_tup[1]]
            writer.writerow(row)
    return True

def main():
     #read config
    config = configparser.ConfigParser()
    config.read('fixer_config.ini')

    # config_dict = {section: dict(config[section]) for section in config.sections()}
    # print(config_dict)
    # for item in config['IO'].items():
    #     print(item)
    
    input_fname = config['IO']['input']
    # xls = pd.ExcelFile(input_fname)
    # print(xls.sheet_names)
    all_sheets = pd.read_excel(input_fname, sheet_name=None)

    anchor_words = set()
    posts = []

    #iterate through tabs
    for sheet_name, df in all_sheets.items():

        #if tab name is weak or none, skip
        if sheet_name == "weak" or sheet_name == "none":
            print(f"Skipping {sheet_name}")
            continue
        elif sheet_name == "universal words":
            print("Processing universal words")
            # print(df.head())
            uni_words = process_universal_words(df)
            anchor_words.update(uni_words)
            print(f"Anchor words: {anchor_words}")            
        else:
            print(f"Processing {sheet_name}")
            new_anchors, tweet_tuples = process_misc_tabs(df)
            posts.extend(tweet_tuples)
            anchor_words.update(new_anchors)
            #extract unigrams, add to shared set of words
            #extract post text and pseudonymized post author, add to shared list of tuples
    
    # print(posts)
    # print(anchor_words)
    print(f"Reading {input_fname}")

    #output anchor words with format <int> <int> <word>
    anchor_fname = config['IO']['anchor_output']
    output_anchors(anchor_words, anchor_fname)
    print(f"Anchor output: {anchor_fname}")
    #output posts with <user hash> <post text>
    post_fname = config['IO']['post_output']
    output_posts(posts, post_fname)
    print(f"Post output: {post_fname}")

    


if __name__ == "__main__":
    main()