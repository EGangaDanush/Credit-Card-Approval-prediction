# decision_tree_credit_card.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

def d_tree(xtrain, xtest, ytrain, ytest):
    """
    Train and evaluate a Decision Tree Classifier
    for Credit Card Approval Prediction.
    """
    # Initialize the Decision Tree model
    dt = DecisionTreeClassifier(random_state=42)

    # Train the model
    dt.fit(xtrain, ytrain)

    # Make predictions on test data
    ypred = dt.predict(xtest)

    # Print evaluation results
    print('*** Decision Tree Classifier ***')
    print('Confusion Matrix:')
    print(confusion_matrix(ytest, ypred))
    print('\nClassification Report:')
    print(classification_report(ytest, ypred))

    return dt, ypred
