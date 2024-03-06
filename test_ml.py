# app.py

import numpy as np
import pandas as pd
from scipy.stats import randint
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
plt.style.use('ggplot')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import classification_report


from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import streamlit as st

#---
credit_risk = pd.read_csv('UCI_Credit_Card.csv')

credit_risk.head()

df = credit_risk.copy()
df.head()

# As we seen Column ID has no meaning here so, we will remove it
df.drop(['ID'], axis = 1, inplace =  True) #axis = 1 -- column removal and inplcae = True --means change in the original data

df.isnull().sum()

df.EDUCATION.value_counts()

df['EDUCATION'].replace({0:1,1:1,2:2,3:3,4:4,5:1,6:1}, inplace = True)
df.EDUCATION.value_counts()

df['MARRIAGE'].value_counts()

df['MARRIAGE'].replace({0:1,1:1,2:2,3:3}, inplace = True)
df['MARRIAGE'].value_counts()

df['PAY_0'].value_counts()

df['default.payment.next.month'].value_counts()

# With column 'limit_bal'
sns.displot(df.LIMIT_BAL, kde = True)

# Independnet features
X = df.drop(['default.payment.next.month'], axis = 1)
# Dependent feature
y = df['default.payment.next.month']
X.head()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train, y_train = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
# print('After oversampling:', Counter(y_train))

logit = LogisticRegression()
logit.fit(X_train, y_train)

# Predicting the model
pred_logit = logit.predict(X_test)

#print("The accuracy of logit model is:", accuracy_score(y_test, pred_logit))
#print(classification_report(y_test, pred_logit))

y_pred = logit.predict(X_test)
pred_class = y_pred

y_test_lr = np.array(y_test)
true_class = y_test_lr

cm1 = metrics.confusion_matrix(true_class, pred_class)

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_predict = rf_clf.predict(X_test)
y_pred2 = rf_clf.predict(X_test)
pred_class = y_pred2
y_test_rf = np.array(y_test)
true_class = y_test_rf
cm2 = metrics.confusion_matrix(true_class, pred_class)


# Streamlit web app
st.title("Credit Risk Classification App")

# Make predictions
st.write(f"Predicted Credit Risk (Logistic Regression):")

accuracy = accuracy_score(y_test_lr, y_pred)
# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2%}")

#---

fig, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(cm1, annot = True, square = True, fmt = 'g')
ax.set_xlabel('Predicted label', fontsize = 15)
ax.set_ylabel('True label', fontsize = 15)
st.pyplot(fig)

#---
st.write(f"Predicted Credit Risk (Decision Tree):")
accuracy2 = accuracy_score(y_test_rf, y_pred2)
st.write(f"Model Accuracy: {accuracy:.2%}")

#---
fig, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(cm2, annot = True, square = True, fmt = 'g')
ax.set_xlabel('Predicted label', fontsize = 15)
ax.set_ylabel('True label', fontsize = 15)
st.pyplot(fig)


