import numpy as np
from sklearn.svm import SVC
from utilities import read_and_prepare_data, encode_categorical_data, split_data, calculate_and_print_metrics

input_file = 'income_data.txt'

X, Y = read_and_prepare_data(input_file)

X_encoded, label_encoder = encode_categorical_data(X)

X = X_encoded[:, :-1].astype(int)
Y = np.array(Y)

X_train, X_test, y_train, y_test = split_data(X, Y)

print("Training model with Gaussian kernel ('rbf')")
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

calculate_and_print_metrics(y_test, y_test_pred)
