import pandas as pd
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

class_counts = df2['target'].value_counts()
total = class_counts.sum()
percentages = (class_counts / total * 100).round(2)

for label, pct in zip(class_counts.index, percentages):
    tumor_type = 'Malignant' if label == 0 else 'Benign'
    print(f"{tumor_type}: {pct}%")

# Calculate imbalance ratio
minority_ratio = class_counts.min() / class_counts.sum()

# Print imbalance info
if minority_ratio < 0.30:
    print(f'Dataset is imbalanced. Minority class is {minority_ratio:.2%} of total.')
else:
    print(f'Dataset is balanced. Minority class is {minority_ratio:.2%} of total.')
