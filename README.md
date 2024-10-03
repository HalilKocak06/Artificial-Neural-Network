# Artificial Neural Network (ANN) for Customer Churn Prediction
This project builds an Artificial Neural Network (ANN) to predict whether a customer will leave a bank based on various customer features. The dataset used includes customer information such as geography, credit score, age, and balance, among others.

## Table of Contents
Installation
Project Structure
Data Preprocessing
Building the ANN
Training the ANN
Prediction
Evaluation
Conclusion
Installation
To run this project, you need the following libraries:

pip install numpy pandas tensorflow scikit-learn
Project Structure
The project is divided into four main parts:

## Data Preprocessing: Encoding categorical data, splitting the dataset, and feature scaling.
Building the ANN: Initializing and constructing the layers of the ANN.
Training the ANN: Compiling and training the model using the training set.
Making Predictions and Evaluating the Model: Testing the model on new data and evaluating performance.
## Data Preprocessing
Importing the Dataset
We use the "Churn_Modelling.csv" dataset. The feature matrix X is derived from columns 3 to the second-to-last column, while the label y is the last column indicating customer churn.

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]
Encoding Categorical Data

## Label Encoding the "Gender" column:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.iloc[:, 2] = le.fit_transform(X.iloc[:, 2])
## One-Hot Encoding the "Geography" column:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
## Splitting the Dataset into Training and Test sets:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Feature Scaling:

python
Copy code
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
## Building the ANN
## Initializing the ANN:

ann = tf.keras.models.Sequential()
Adding Layers:

## Input layer and two hidden layers:
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
Output layer (binary classification):
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
## Training the ANN
Compiling the ANN:

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## Training the model:


ann.fit(X_train, y_train, batch_size=32, epochs=100)
Prediction
Predicting a Single Observation
To predict if a specific customer will leave the bank based on their features:

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
Example Input:

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60,000
Number of Products: 2
Credit Card: Yes
Active Member: Yes
Estimated Salary: $50,000
Model Output: The ANN model predicts that this customer will not leave the bank.

Note: The input format for the prediction must be a 2D array, and one-hot encoded variables (like "Geography") should be handled properly in the input.

Evaluation
Predicting the Test Set Results:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
Confusion Matrix and Accuracy:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
Conclusion
This project builds a simple ANN model to predict customer churn with a binary classification output. By utilizing data preprocessing techniques and leveraging TensorFlow's Keras API, the model effectively predicts whether a customer will leave based on their profile.

Feel free to test this model on new datasets and fine-tune it by experimenting with the ANN structure or hyperparameters!
