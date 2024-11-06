import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data.txt')

data = data.dropna(subset=['price'])

data['origin_enc'] = LabelEncoder().fit_transform(data['origin'])
data['destination_enc'] = LabelEncoder().fit_transform(data['destination'])
data['train_type_enc'] = LabelEncoder().fit_transform(data['train_type'])
data['train_class_enc'] = LabelEncoder().fit_transform(data['train_class'])
data['fare_enc'] = LabelEncoder().fit_transform(data['fare'])

X = data[['origin_enc', 'destination_enc', 'train_type_enc', 'train_class_enc', 'fare_enc']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BayesianRidge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Середньоквадратична похибка на тестовому наборі: {mse}")

def predict_price(origin, destination, train_type, train_class, fare):
    origin_enc = LabelEncoder().fit(data['origin']).transform([origin])[0]
    destination_enc = LabelEncoder().fit(data['destination']).transform([destination])[0]
    train_type_enc = LabelEncoder().fit(data['train_type']).transform([train_type])[0]
    train_class_enc = LabelEncoder().fit(data['train_class']).transform([train_class])[0]
    fare_enc = LabelEncoder().fit(data['fare']).transform([fare])[0]
    
    input_data = pd.DataFrame([[origin_enc, destination_enc, train_type_enc, train_class_enc, fare_enc]],
                              columns=['origin_enc', 'destination_enc', 'train_type_enc', 'train_class_enc', 'fare_enc'])
    
    predicted_price = model.predict(input_data)[0]
    return predicted_price

origin = input("Enter the origin station: ")
destination = input("Enter the destination station: ")
train_type = input("Enter the train type (e.g., AVE): ")
train_class = input("Enter the train class (e.g., Turista): ")
fare = input("Enter the fare type (e.g., Promo): ")

predicted_price = predict_price(origin, destination, train_type, train_class, fare)
print(f"The predicted price is: {predicted_price}")