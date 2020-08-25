import random
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier




train = open("train.txt", "r+", encoding="utf8").read().split("\n")


cv = CountVectorizer(binary=True)
cv.fit(train)
X = cv.transform(train)
#print(X)
#3665

target = [1 if i < (len(train)/2) else 0 for i in range(len(train))]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size=0.75
)

#LOGISTIC REGRESSION
'''for c in [0.01, 0.05, 0.25, 0.5, 1]:
    clf = LogisticRegression(C=c)
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))'''

#SUPPORT VECTOR MACHINES
'''for c in [0.01, 0.05, 0.25, 0.5, 1]:
    clf = svm.SVC(gamma=c, kernel='sigmoid')
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, clf.predict(X_val))))'''

#K NEAREST NEIGHBOURS
'''for c in [1, 2, 3, 4, 5]:
    clf = KNeighborsRegressor(n_neighbors=c, algorithm='brute')
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, clf.score(y_val, clf.predict(X_val))))'''

#NAIVE BAYES
'''for c in [0.01, 0.05, 0.25, 0.5, 1]:
    clf = MultinomialNB(alpha=c, class_prior=None, fit_prior=True)
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, clf.predict(X_val))))'''

#DECISION TREES
'''for c in [10, 50, 200, 400, None]:
    clf = DecisionTreeClassifier(max_depth=c, splitter='random', criterion='gini')
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, clf.predict(X_val))))'''

#NEURAL NETWORKS
for c in [2, 3, 4]:
    clf = MLPClassifier(activation='tanh', solver='sgd', alpha=0.0001, hidden_layer_sizes=(c, 2), learning_rate='adaptive', learning_rate_init=0.001)
    clf.fit(X_train, y_train)
    print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, clf.predict(X_val))))


test = open("test.txt", "r+", encoding="utf8").read().split("\n")
X_test = cv.transform(test)
result = clf.predict(X_test)

#probabilty of positive
print('Positive probability', result)


#X_test = cv.transform(reviews_test_clean)

#neuTexts = open("neu.txt", "r+", encoding="utf8").read().split("\n")


