# Author: Anshit Saxena - 16292998
# Python script to print alternate characters of a string

# function to create a new string with alternate characters from the original string
def string_alternative(s):
    newS = slice(0,len(s),2)
    return(s[newS])

# Ask user to enter any sentence
oString = str(input("Please type a sentence: "))

# Call the  string_alternative function and show the the final result
print("The sentence with alternate characters : ")
print(string_alternative(oString))