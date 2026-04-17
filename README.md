# DP-WAG

Given list of anchor words, and social media post data, generate differentially private
word association graph and extract/output top k edges, detected communities, distribution of weights, visualizations.

## Description

This script is intended to be a proof-of-concept for differentially private
word association graph generation and community detection. This script 
was developed iteratively and needs work to make it more organized, readable,
and usable.

There are two modes of co-occurrence used to generate these matrices: full-post and adjacency. Full-post counts co-occurrence when any two anchor words appear in the same post at all. Adjacency requires them to be next to one another. 


## Usage
### Example: run on reddit data with partitions ignoring edges under 20.0, with adjacency flow. Ignore indices flag recommended for now. Requires both a list of anchor words and posts
```
python dp-wag.py anchor_words_reddit_privacy_week_20260223.txt reddit_privacy_week_20260223_tweet_texts.hashed.txt --out reddit_topk.csv --out_partition reddit_partitions.csv --partition_threshold 20.0 --adjacency --ignore_indices
```

## Examples
### Examples directory contains sample outputs from the script on the reddit dataset with a threshold of 20.0, using adjacency co-occurrence.