from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf

def FitLogisticRegression(X, y):
    # Create an instance of LogisticRegression
    logisticRegressionModel = LogisticRegression()

    # Fit the model on the generated data
    logisticRegressionModel.fit(X, y)

    coefficients = logisticRegressionModel.coef_
    bias = logisticRegressionModel.intercept_

    # Combine coefficients and bias into a single vector
    learnedParameters = np.concatenate((bias, coefficients[0]))

    print("Learned parameters")
    print(learnedParameters)
    return learnedParameters.tolist()

def FitLogisticRegressionTF(X, y):

    # Convert data to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Build a logistic regression model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=3000, verbose = 0)

    # Get the learned parameters
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]
    learnedParameters = np.concatenate((bias, weights.reshape(-1)))

    print("Learned parameters TF")
    print(learnedParameters)

    return learnedParameters.tolist()

# ====================================================================

# https://www.datacamp.com/tutorial/understanding-logistic-regression-python

##import pandas
#import pandas as pd
#import os
#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
## load dataset
#print(os.getcwd())
#pima = pd.read_csv("./extras/diabetes.csv", header=None, names=col_names)

#pima.head()

##split dataset in features and target variable
#feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
#X = pima[feature_cols] # Features
#y = pima.label # Target variable

## split X and y into training and testing sets
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

## import the class
#from sklearn.linear_model import LogisticRegression

## instantiate the model (using the default parameters)
#logreg = LogisticRegression(random_state=16)

## fit the model with data
#logreg.fit(X_train, y_train)

#y_pred = logreg.predict(X_test)

## import the metrics class
#from sklearn import metrics

#cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#cnf_matrix


## import required modules
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

#class_names=[0,1] # name  of classes
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)
## create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')

#plt.text(0.5,257.44,'Predicted label');

#from sklearn.metrics import classification_report
#target_names = ['without diabetes', 'with diabetes']
#print(classification_report(y_test, y_pred, target_names=target_names))

#y_pred_proba = logreg.predict_proba(X_test)[::,1]
#fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
#auc = metrics.roc_auc_score(y_test, y_pred_proba)
#plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()