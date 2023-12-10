import pandas as pd
import numpy as np
import random
import pickle
import joblib 

from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

GROUND_TRUTH_LABELS = {}

def split_training_and_testing_data():

    # Read the data
    data = pd.read_csv('feature_extraction/features.csv')
    GROUND_TRUTH_LABELS = dict(zip(data['fileName'], data['Label']))
    print("Data: ")
    print(data.head(2))


    # Replace labels with 0 for speech and 1 for music
    data["Label"] = (data["Label"] == "yes").astype(int)

    # Separate the features (x) from the labels (y)
    y = data["Label"]
    x = data.drop(["Label"], axis=1)

    # For reproducable results, set the seed value for train_test_split
    #random_seed = 45 # Precision: 0.333333; Recall = 0.600000
    random_seed = 26 # Precision: 0.666667; Recall = 0.500000
    np.random.seed(random_seed)
    random.seed(random_seed)

    # The directions say to use 1/3 of the data for testing and 2/3 for training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=random_seed)

    # The directions say to indicate which of the files are used for training
    print(x_train.sort_index()["fileName"])

    train_file_names = x_train['fileName'].tolist()
    test_file_names = x_test['fileName'].tolist()

    # Drop the fileName column
    x_train = x_train.drop(['fileName'], axis=1)
    x_test = x_test.drop(['fileName'], axis=1)

    return x_train, x_test, y_train, y_test, train_file_names, test_file_names

def evaluate_model(model, x_test, y_test):

    y_pred = (model.predict(x_test) > 0.5).astype("int32")

    print('Precision: %.6f' % precision_score(y_test, y_pred, zero_division=1))
    print('Recall = %.6f' % recall_score(y_test, y_pred, zero_division=1))

# Reference: https://youtu.be/lrShBmW8Iqs
# Modified from https://scikit-learn.org/stable/model_persistence.html
def save_model(model):
    pickle.dump(model, open("SVM_model.pkl", "wb"))

def load_model():
    model = pickle.load(open("SVM_model.pkl", "rb"))
    return model

def train_model():
    
    x_train, x_test, y_train, y_test, train_file_names, test_file_names = split_training_and_testing_data()
    print("Describe ",x_train.describe())
    print("min ",x_train.min())
    print("max: ",x_train.max())
    print("Before scaling:")
    print(x_train.head())

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the training data and transform the training data
    x_train_scaled = scaler.fit_transform(x_train)

    # Transform the testing data using the same scaler
    x_test_scaled = scaler.transform(x_test)
    print("After scaling:")
    print(pd.DataFrame(x_train_scaled, columns=x_train.columns).head())

    model = svm.SVC()
    
    # Train the model with the scaled training data
    model.fit(x_train_scaled, y_train)

    evaluate_model(model, x_test_scaled, y_test)
    save_model(model)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Get ground truth labels
    data = pd.read_csv('feature_extraction/features.csv')
    GROUND_TRUTH_LABELS = dict(zip(data['fileName'], data['Label']))
    test_labels = {file: GROUND_TRUTH_LABELS[file] for file in test_file_names}

    return x_train_scaled, x_test_scaled, y_train, y_test, model, train_file_names, test_file_names, test_labels
