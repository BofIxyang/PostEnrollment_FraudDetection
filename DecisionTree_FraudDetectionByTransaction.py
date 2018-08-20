from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV  # got confliction 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

#from imblearn.over_sampling import SMOTE


df = pd.read_csv("X:\Data Analytics\Fraud Analysis\DecisionTree_DetectionByTransaction\output8.csv", index_col=0)
df = df.drop (['OpenDate','firstFundingDate'], axis=1)

EmailDigitCount = []
for row in df['Email1']:
    EmailDigitCount.append(sum(c.isdigit() for c in row))
df['EmailDigitCount'] = EmailDigitCount
df = df.drop(['Email1'], axis=1)

# coding the string values
df = pd.get_dummies(df, prefix=['FundingTransDesc','AccountType'], columns = ['FundingTransDesc','AccountType'])



# Replace missing values with median
imp = Imputer(missing_values = np.nan, strategy='median', axis=1)
imputed_DF = pd.DataFrame(imp.fit_transform(df))
imputed_DF.columns = df.columns
imputed_DF.index = df.index

# Verify there is no NA or inf value
np.any(np.isnan(imputed_DF))
np.all(np.isfinite(imputed_DF))


# Prepare training and testing data
X = imputed_DF.drop('fraudsterFlag', axis=1)
y = imputed_DF["fraudsterFlag"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
sum(y_test)


# Try over/under weight sampling
#sm = SMOTE(random_state=12, ratio = 1.0)
#X_train_res, y_train_res = sm.fit_sample((X_train, y_train)



# Try adjusting postive/negative weight
minority_wt = 0.985
model = DecisionTreeClassifier(criterion='entropy', max_depth = 4, class_weight={0:1-minority_wt, 1:minority_wt})


# CV
parameters = {'max_depth':range(3,5)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)
clf.fit(X=X_train, y=y_train)
tree_model = clf.best_estimator
print(clf.beset_score_, clf.best_params_)


model.fit(X_train,y_train)
model.score(X_train,y_train)

Y_pred = model.predict(X_test)

#accuracy_score(y_test, Y_pred)*100

# Verify confusin matrix
confusion_matrix(y_test, Y_pred, labels=[0, 1])


# Plot the tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


import graphviz
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("test")








FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)





