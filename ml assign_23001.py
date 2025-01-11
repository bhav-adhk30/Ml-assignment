import numpy as np
import matplotlib.pyplot as plt
# question 1
import random
import statistics
v1=[]
for i in range(100):
    v1.append(random.randint(0,100))
v1.sort()
print(v1)
print('*'*150)

#qn2

multiplied=v1*3
print(multiplied) #after multiplying, the list has been repeated 3 times
print('*'*150)

#qn3
mean=statistics.mean(v1)
stand_dev=statistics.stdev(v1)
print(mean)
print(stand_dev)
print('*'*150)

#qn4
matrix=[]
for i in range(4):
    row=[]
    for j in range(3):
        row.append(0)
    matrix.append(row)

for i in range(4):
    for j in range(3):
        matrix[i][j]=random.randint(0,100)

for row in matrix:
    print(row)

flattened=[]
for row in matrix:
    for num in row:
        flattened.append(num)

print("the flattened array is",flattened)
print('*'*150)

#qn5
s1 = " I am a great learner. I am going to have an awesome life "
words=s1.split()
count=0

for i in words:
    if i=="am":
        count= count+1
        
print("the number of occurences are", count)
print("*"*120)

#qn6
s2="I work hard and shall be rewarded well"
print(s1+s2)


#qn7
S3=s1+s2
S3_modified = S3.replace('.', ' ')

words = S3_modified.split()
length = len(words)

print("Words in the array:", words)
print("Length of the array:", length)
print('*'*130)


#qn8
new_arr=[]
words_to_del=['I',"am","to","and"]
for i in words:
    if i not in words_to_del and len(i)<=6:
        new_arr.append(i)

print("the updated array is:", new_arr)
print("length of updated array is:", len(new_arr))
print('*'*130)

#qn9
given_date="01-JUN-2021"
day, month, year= given_date.split('-')
print("original date:",day, month, year)

mapping= {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

numerical= mapping[month]
print("day is", day)
print("month is", numerical)
print("yr is:", year)
print('*'*130)


#q10
import pandas as pd

data = {
    "City": [
        "BENGALURU", "CHENNAI", "MUMBAI", "MYSURU", "PATNA",
        "JAMMU", "GANDHI NAGAR", "HYDERABAD", "ERNAKULAM", "AMARAVATI"
    ],
    "State": ["KA", "TN", "MH", "KA", "BH", "JK", "GJ", "TS", "KL", "AP"],
    "PIN Code": [560001, 600001, 400001, 570001, 800001, 180001, 382001, 500001, 682001, 522001]
}

df = pd.DataFrame(data)

df["City, State"] = df["City"] + ", " + df["State"]

df.to_excel("cities_with_states.xlsx", index=False, engine="openpyxl")

print(df)

print('*'*130)

#q11
plt.plot(v1, color='blue')
plt.title("Sorted Vector V1")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
print('*'*130)

#q12
V2 = [x**2 for x in v1]
plt.plot(v1, label="V1 (Sorted)", color='red')  
plt.plot(np.sort(V2), label="V2 (Squared)", color='blue')
plt.title("Plot of V1 and V2")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
