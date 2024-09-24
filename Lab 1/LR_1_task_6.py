import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('data_multivar_nb.txt', header=None)
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)

print("SVM - Матриця плутанини:")
print(confusion_matrix(y_test, svm_y_pred))
print("\nSVM - Звіт про класифікацію:")
print(classification_report(y_test, svm_y_pred))

print("Naive Bayes - Матриця плутанини:")
print(confusion_matrix(y_test, nb_y_pred))
print("\nNaive Bayes - Звіт про класифікацію:")
print(classification_report(y_test, nb_y_pred))
