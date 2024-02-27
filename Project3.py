import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('wdbc.data.mb.csv', header=None)

# Split the dataset into features (X) and labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def calculate_class_probabilities(X_train, y_train):
    # Calculate class probabilities
    classes = np.unique(y_train)
    probabilities = {}
    
    for cls in classes:
        # Select data for the current class
        X_cls = X_train[y_train == cls]
        
        # Calculate mean and standard deviation for continuous attributes
        means = X_cls.mean(axis=0)
        stds = X_cls.std(axis=0)
        
        # Store the results
        probabilities[cls] = (means, stds)
    
    return probabilities

def predict(X_test, probabilities):
    predictions = []
    
    for i in range(X_test.shape[0]):
        sample = X_test.iloc[i]
        max_prob = -1
        predicted_class = None
        
        for cls, (means, stds) in probabilities.items():
            # Calculate the class likelihood
            likelihood = np.exp(-0.5 * ((sample - means) / stds)**2) / (np.sqrt(2 * np.pi) * stds)
            class_prob = np.prod(likelihood)
            
            # Update the predicted class if this class has higher probability
            if class_prob > max_prob:
                max_prob = class_prob
                predicted_class = cls
        
        predictions.append(predicted_class)
    
    return predictions

# Calculate class probabilities on the training set
probabilities = calculate_class_probabilities(X_train, y_train)

# Make predictions on the test set
y_pred = predict(X_test, probabilities)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate and display the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

# Define the number of folds (K)
k = 5
skf = StratifiedKFold(n_splits=k)

# Initialize variables to store accuracy and confusion matrices
accuracies = []
confusion_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Calculate class probabilities on the training set
    probabilities = calculate_class_probabilities(X_train, y_train)

    # Make predictions on the test set
    y_pred = predict(X_test, probabilities)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calculate and store the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(confusion)

# Display accuracy for each fold
for i, acc in enumerate(accuracies):
    print(f'Fold {i+1} Accuracy: {acc}')

# Display confusion matrices for each fold
for i, conf in enumerate(confusion_matrices):
    print(f'Fold {i+1} Confusion Matrix:')
    print(confusion)
