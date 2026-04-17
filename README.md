# DP-WAG

Given list of anchor words, and social media post data, generate differentially private
word association graph and extract/output top k edges, detected communities, distribution of weights, visualizations.

## Description

This script is intended to be a proof-of-concept for differentially private
word association graph generation and community detection. This script 
was developed iteratively and needs work to make it more organized, readable,
and usable.

## Usage
### Example: run on reddit data with partitions ignoring edges under 20.0, with adjacency flow. Ignore indices flag recommended for now.
```
python dp-wag.py reddit_privacy_week_20260223.txt reddit_privacy_week_20260223_tweet_texts.hashed.txt --out reddit_topk.csv --out_partition reddit_partitions.csv --partition_threshold 20.0 --adjacency --ignore_indices
```