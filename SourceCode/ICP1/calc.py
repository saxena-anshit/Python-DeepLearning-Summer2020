# Author: Anshit Saxena - 16292998
# Python script to perform arithmetic operations on 2 numbers taken through user input

# Function to perform arithmetic operations (addition, subtraction, multiplication, and division)
def mathOperations(number1, number2):
    print("Addition of " + str(number1) + " & " + str(number2) + " is: " + str(number1 + number2))
    print("Subtraction of " + str(number1) + " & " + str(number2) + " is: " + str(number1 - number2))
    print("Multiplication of " + str(number1) + " & " + str(number2) + " is: " + str(number1 * number2))
    print("Division of " + str(number1) + " & " + str(number2) + " is: " + str(number1 / number2))

#To ask the user to enter two numbers

n1 = float(input("Please Enter the first number: "))
n2 = float(input("Please Enter the second number: "))

#To call the mathOperations function
mathOperations(n1,n2)