# diabetes
logistic reg model
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
data = pd.read_csv('diabetes2.csv', header = 0)
print(data.head(5))
print(data.tail(5))
data.info()
data.corr()
Y=data['Outcome']
X=data.drop('Outcome',axis=1)
X = pd.get_dummies(X, drop_first = True)
X.head()
sns.pairplot(data,kind='reg')
plt.show()
sns.heatmap(data.corr())
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7 , test_size=0.3, random_state=10)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver = 'liblinear', class_weight="balanced")
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score,precision_score,recall_score, roc_curve, auc, roc_auc_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy is  :' ,accuracy)
print('Precision is  :',precision)
print('Recall is  :',recall)
print('Roc Auc is  :',roc_auc)
print('Confusion Matrix is  :\n',cm)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0,1],[0,1],'y--')
#plt.xlim([0,1])
#plt.ylim([0,1])
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.xlabel('False Positive Rate(1-Specificity)')
plt.show()

print("Area under the curve: ",round(auc(false_positive_rate, true_positive_rate),3))
