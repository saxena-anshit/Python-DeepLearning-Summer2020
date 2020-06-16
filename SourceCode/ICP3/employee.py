# Author: Anshit Saxena - 16292998
# Python script to create classes, inheritance of classes and using constructors

class Employee():
    # for counting the number of employees, and for adding salaries and names.
    empNum, empName, empFam, empSal, empDep = 0, [], [], [], [];

    # default constructor for the class to initialize
    def __init__(self, name, family, salary, department):
        self.empname = name  # Storing the name in self.empname
        self.empfamily = family  # Storing the family in self.empfamily
        self.empsalary = salary  # Storing the salary in self.empsalary
        self.empdepartment = department  # Storing the department in self.empdepartment
        Employee.empNum = Employee.empNum + 1  # Counting the employees' numbers
        Employee.empName.append(name)  # appends name attribute
        Employee.empFam.append(family)  # appends family attribute
        Employee.empSal.append(salary)  # appends salaray attribute
        Employee.empDep.append(department)  # appends department attribute
    # function to calculate the average salary
    def average(self):
        sumSal = 0;
        for i in Employee.empSal:
            sumSal += int(i);
        return sumSal / len(Employee.empName)


class FullEmployee(Employee):
    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)


fe1 = FullEmployee('Ansh', 'Saxena', '10000', 'Computer Science')
fe2 = Employee('ABC', 'XYZ', '20000', 'Software Engineering')
# fe3 = FullEmployee('Bill', 'Gates', '330000000', 'No Idea, Maybe Microsoft')

print("The number of employees is: ", fe1.empNum)
print("The employees' names are: ", fe1.empName)
print("The employees' families are: ", fe1.empFam)
print("The employees' salaries are:", fe1.empSal)
print("The employees' departments are:", fe1.empDep)
print('The average salary is', FullEmployee.average(Employee))