import pandas as pd
from collections import Counter
from itertools import combinations



def get_word_set(input_fname):
    df = pd.read_csv(input_fname, sep='\t')

    word_set = set()
    for cell in df['Story hook']:
        words = [w.strip() for w in cell.split(",")]
        word_set.update(words)

    return word_set

def get_word_counts(input_fname):
    df = pd.read_csv(input_fname, sep='\t')

    word_counts = Counter()
    for cell in df['Story hook']:
        words = [w.strip() for w in cell.split(",")]
        word_counts.update(words)

    # print(word_counts)
    return word_counts

#TODO probably better to make this a 2d matrix for easier lookups on either word
def story_hook_cooccurrences(input_fname):
    df = pd.read_csv(input_fname, sep='\t')

    co_counts = Counter()
    for cell in df['Story hook']:
        words = [w.strip() for w in cell.split(",")]
        for pair in combinations(sorted(words), 2):
            co_counts[pair] += 1
    
    # co_counts = dict(co_counts)
    return co_counts

def write_output(co_ocs, out_fname):
    co_df = pd.DataFrame(
        [(pair[0], pair[1], count) for pair, count in co_ocs.most_common()],
        columns=['Word1', 'Word2', 'Count']
    )
    co_df.to_csv(out_fname, sep='\t')

def main():
    #TODO make configurable
    # input_fname = "outputs_914/shared_partitions_win1.tsv"
    # output_fname = "outputs_914/storyhook_cooc_win1.tsv" 
    input_fname = "Girvan-Newman_outputs/winF/shared_partitions_winF.tsv"
    output_fname = "Girvan-Newman_outputs/winF/storyhook_cooc_winF.tsv" 


    # story_words = get_word_set(input_fname)
    word_counts = get_word_counts(input_fname)

    co_ocs = story_hook_cooccurrences(input_fname)
    # for pair, count in co_ocs.most_common(10):
    #     print(pair, count)

    write_output(co_ocs, output_fname)


    return 0

if __name__ == "__main__":
    main()