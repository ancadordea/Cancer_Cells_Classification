import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Data processing
data = load_breast_cancer()

# Splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)
print('Train set:', x_train.shape,  y_train.shape)
print('Test set:', x_test.shape,  y_test.shape)

# Training the model
model = GaussianNB()
model.fit(x_train, y_train)

# Predicting tumour classifications
y_pred = model.predict(x_test)
print(y_pred[:100])

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()