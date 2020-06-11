# Author: Anshit Saxena - 16292998
# Python script to count the number of times a word occurs in a file
# and write the output in a file

# function to count the words using dict and split methods
def count(filewords):
    counts = dict()
    words = filewords.split()

# counting the words after split
    for w in words:
        if w in counts:
            counts[w] += 1
        else:
            counts[w] = 1
    return counts

# reading the text file
f = open("text", "r")
words = f.read()

# calling the counting fucntion
result = count(words)

# writing the output in a file test.txt
file = open("test.txt", "a")
for r, v in result.items():
    file.write("\n" + r + " = "+ str(v) + "\n")