import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df=pd.DataFrame(data.data,columns=data.feature_names)

# Display a random sample of rows
print('Random records from df =')
print(df.sample(5))

# Column data types
print('Dataset info =')
print(df.info())

# Summary statistics of data
print('Dataset summary =')
print(df.describe())

# Distribution of malignant and benign 
print('Distribution of malignant and benign =')
df2 = pd.DataFrame(data.target,columns=['target'])
print(df2.sample(5))

# Count the classes
class_counts=df2['target'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.2f%%', colors=['pink', 'purple'])
plt.show()

# Calculate imbalance ratio
minority_ratio = class_counts.min() / class_counts.sum()

# Print imbalance info
if minority_ratio < 0.30:
    print(f'Dataset is imbalanced. Minority class is {minority_ratio:.2%} of total.')
else:
    print(f'Dataset is balanced. Minority class is {minority_ratio:.2%} of total.')
