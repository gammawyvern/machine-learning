import numpy as np;
import pandas as pd;

array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]);

# Question 1
mask = array < 5;
print(mask);
print(array[mask]);
print();

# Question 2
print(array[array % 2 == 0]);
print();

# Question 3
print(array_2d[array_2d > 5]);
print(array_2d[(array_2d > 2) & (array_2d < 6)]);
print();

raw_data = {
    "age":      [ 15,  20,  25,  30,  35],
    "height":   [150, 170, 180, 160, 175],
    "weight":   [ 50,  60,  70,  80,  75],
}
data_frame = pd.DataFrame(raw_data);

print(data_frame[data_frame["age"] > 20])
print(data_frame[(data_frame["age"] >= 20) & (data_frame["height"] > 160)])

students = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'major': ['Computer Science', 'Physics', 'Math', 'Computer Science', 'Biology'],
    'GPA': [3.2, 3.7, 3.5, 2.9, 3.8]
})
mask = (students['GPA'] > 3.0) & (students["major"] == "Computer Science");
print(students[mask]);
