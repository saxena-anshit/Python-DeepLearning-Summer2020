# Author: Anshit Saxena - 16292998
# Python script to get weights in lbs and convert into kgs

# Take user input of the number of students
num = int(input("Enter the number of students: "))

# Creating empty lists for lbs and kgs
lb = []
kg = []

# Loop to append weights one by one to lbs list through user input
for i in range(num):
    x = float(input("Enter a weight in lbs  >> "))
    lb.append(x)

# Conversion from lbs to kgs
kg = [round(x / 2.2, 2) for x in (lb)]


# Printing the Weights
print(lb)
print(kg)