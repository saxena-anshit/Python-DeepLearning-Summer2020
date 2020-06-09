# Author: Anshit Saxena - 16292998
# Python script to delete 2 characters and reverse the result

# import 'randrange' to generate a random number
from random import randrange

# function to delete 2 characters and call the the reverse function
def deleteFromOriginal(string):
    # getting a random number so that random removal can be performed
    x = randrange(len(string)) - 2
    # To show the first 'x' characters in the word and from the 'x+2' characters to the end
    string = string[0:x] + string[x+2:]
    return (string)

# Function to reverse a string
def reverseString(string):
    string = string[::-1]
    return string

# Ask user to enter any word
orginalString = str(input("Please type any word: "))

# Call the delete function and stor the result in new variable
afterDelete = deleteFromOriginal(orginalString)

# Call the rverse function and show the the final result
print("The string after delete 2 characters and reverse it: ")
print(reverseString(afterDelete))