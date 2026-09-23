# DP-WAG

Given list of anchor words, and social media post data, generate differentially private word association graph and extract/output top k edges, detected communities, distribution of weights, visualizations.

## Description

This script is intended to be a proof-of-concept for differentially private
word association graph generation and community detection. This script 
was developed iteratively and needs work to make it more organized, readable, and usable.

## Config

The config.ini file controls the internal variables for the script to run. Input and output files can be whatever you'd like, as long as the inputs conform to the output from the [wordchipper](https://chattersum.com/wordchipper/). The dataset_times variable takes the form of a json dictionary whose keys are communities and whose values are a timeframe string.

The shared_partition_fname variable appends community data to a file intended to be shared across runs, for ease of comparitive analysis. This was not intended as a long-term solution. 

The threshold variable in the partition section is the minimum edge weight to be considering in partition detection.

The cooccurrence_window variable in the statistics section determines how close anchor words must be in order to be counted. Setting this to -1 will make the full post the co-occurrence window. 

## Usage
First, set any appropriate variables in the config.ini file. Then,

```
python dp-wag.py
```