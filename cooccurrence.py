#OUTLINE

#process anchor word dataset
    #extract word_index and word
    #create hashmap of word to word_index

#process post dataset
    #hashmap, key = user hash, value = post text
    #for each post, create a list for each anchor word
    #count occurences of anchor words in a post in the list
    #create per post co-occurence matrix (how?)

#create hashmap from user hash to matrix

#for each post
    #create_or_get(user hash) matrix  m
    #add counts of post matrix to m

#for each user
    #write matrix to output file 

def main():
    print("Hello world")

if __name__ == "__main__":
    main()
