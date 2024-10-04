import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_and_prepare_data(input_file, max_datapoints=25000):
    X = []
    Y = []
    count_class1 = 0
    count_class2 = 0

    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue

            data = line[:-1].split(', ')

            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                Y.append(0)
                count_class1 += 1
            elif data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                Y.append(1)
                count_class2 += 1

    X = np.array(X)
    print(f"Data successfully read.\nShape of array X: {X.shape}")
    return X, np.array(Y)


def encode_categorical_data(X):
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            le = preprocessing.LabelEncoder()
            X_encoded[:, i] = le.fit_transform(X[:, i])
            label_encoder.append(le)
    
    print(f"Encoding completed.\nShape of X after encoding: {X_encoded.shape}")
    return X_encoded, label_encoder


def split_data(X, Y, test_size=0.2, random_state=5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print("Data successfully split into training and testing sets.")
    return X_train, X_test, y_train, y_test


def calculate_and_print_metrics(y_test, y_test_pred):
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")