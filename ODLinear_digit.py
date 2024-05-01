"""
   Test the One-vs-the-Rest classification routine with Orthogonal Distance 
   Regression
   
   The Digit dataset is used.
   
   Authors : scikit-learn example authors
             Wing-Fai Thi
   
   Licence : GNU v 3.0
   
   History : 14/3/2018
             1/5/2024
"""

# Digit (multi-class) classification with ODLogisticRegressionOVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn import metrics 
from ODLinear import *

t0 = time.time()

digits = load_digits()

X = digits.data
y = digits.target

# figure size in inches
fig = plt.figure(figsize=(6, 6)) 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[]) 
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest') 
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.savefig('digit_examples.png')

# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,test_size=0.2,random_state=1)
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# train the model
clf = OrthogonalDistanceLogisticRegressionOVR(C=100,tol=1e-4)

clf.fit(X_train, y_train)

# use the model to predict the labels of the test data
predicted = clf.predict(X_test)
probability = clf.predict_proba(X_test)[:,0]
expected = y_test

# Plot the prediction
fig = plt.figure(figsize=(6, 6)) # figure size in inches 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[]) 
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')
    # label the image with the target value
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(predicted[i]), color='red')

plt.savefig('digit_predictions.png')

matches = (predicted == expected)
print("Match:",matches.sum())
print("Sample size:",len(matches))
print("Match/total:", matches.sum() / float(len(matches)))
print(metrics.classification_report(expected, predicted))

cm = metrics.confusion_matrix(expected, predicted)
print('Confusion matrix')
print(cm)

plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x),
                     horizontalalignment='center',
                     verticalalignment='center')
plt.colorbar()
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.05)
plt.savefig('digit_confusion_matrix.png')

train_score = clf.score(X_train, y_train)
print("Training set score: %f" % train_score)
test_score = clf.score(X_test, y_test)
print("Test set score: %f" % test_score)

sparsity = np.mean(abs(clf.coef_) <  1e-2) * 100
print("Sparsity with : %.2f%%" % sparsity)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()

for i in range(10):
    odr_plot = plt.subplot(2, 5, i + 1)
    odr_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                     cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    odr_plot.set_xticks(())
    odr_plot.set_yticks(())
    odr_plot.set_xlabel('Class %i' % i)
    plt.suptitle('Classification vector for...')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.savefig('digit_classification_vectors.png')
plt.show()
