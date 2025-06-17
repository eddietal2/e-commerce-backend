# Refreshing on some Python
import sys
import random

print("Hello, World!")

# Python Version
print(sys.version)

if(5 > 3):
    print('Winner winner chicken dinner')

# Variables
x = 5
y = 8
print(x+y)
a, b, c = 'A', 'B', 'C'
print(a)
print(b)
print(c)
fruits = ['apples', 'oranges', 'bananas']
d, e, f = fruits
print(d)
print(e)
print(f)

print(a, b, c, d, e, f, x, y)

# Casting
name = str('Eddie Taliaferro II')
print(type(name))
print(name)

# Global + Function Declaration
x = 'awesome'

def myFunc():
    global x
    x = 'fantastic'

myFunc()
print('Global is ' + x)

# Data Type
floaty = 2.8
print(int(floaty))
print('Random Number: ', random.randrange(1,10))

# BOOKMARK
# https://www.w3schools.com/python/python_casting.asp
