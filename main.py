import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dispkernel import dispKernel

import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
from torch.utils.data import DataLoader
!pip install model 
from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

import scipy.signal as signal

seed = 1
torch.manual_seed(seed)

# =================================== LOAD DATASET =========================================== #

data = pd.read_csv('adult.csv')

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

print("shape",data.shape)
print("columns",data.columns)
print("head",data.head())
print("value counts",data["income"].value_counts())

# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
num_missing_entries = 0
for feature in col_names:
    print("missing value count ",feature, data[feature].isin(["?"]).sum())

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error
print("data[data[income] != >50K]", data[data["income"] != ">50K"])

index_occupation = data[data["occupation"] == "?"].index
data.drop(index_occupation, inplace=True)
print("index_occupation",index_occupation)
index_nativec = data[data["native-country"] == "?"].index
print("index_nativec",index_nativec)
data.drop(index_nativec, inplace=True)

# =================================== BALANCE DATASET =========================================== #

min_income_class_count = min(data["income"].value_counts())
greater_income_rows = data.loc[data["income"] == ">50K"].sample(n=min_income_class_count, random_state=seed)
less_income_rows = data.loc[data["income"] == "<=50K"].sample(n=min_income_class_count, random_state=seed)
balanced_data = pd.concat([greater_income_rows, less_income_rows])

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

verbose_print(balanced_data.describe())

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    print(balanced_data[feature].value_counts())

# visualize the first 3 features using pie and bar graphs

# for i in range(3):
#     pie_chart(balanced_data, categorical_feats[i])
#     binary_bar_chart(balanced_data, categorical_feats[i])

# for feature in categorical_feats:
#     pie_chart(data, feature)
#     binary_bar_chart(data, feature)

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

continuous_feats = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss",
                    "hours-per-week"]

# NORMALIZE CONTINUOUS FEATURES
continuous_data = balanced_data[continuous_feats].copy()
continuous_data_stats = continuous_data.describe()
means = continuous_data_stats.loc["mean"]
stds = continuous_data_stats.loc["std"]
continuous_data -= means
continuous_data /= stds
continuous_data = continuous_data.to_numpy()

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
encoded_data = balanced_data[categorical_feats].apply(lambda col: label_encoder.fit_transform(col)).copy()
income_col = encoded_data["income"].copy().to_numpy()
del encoded_data["income"]

oneh_encoder = OneHotEncoder(categories="auto")
encoded_data = oneh_encoder.fit_transform(encoded_data).toarray()

# STITCH CONTINUOUS AND CATEGORICAL FEATURES TOGETHER
processed_data = np.concatenate((encoded_data, continuous_data), axis=1)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

training_data, validation_data, training_labels, validation_labels = train_test_split(processed_data, income_col,
                                                                                      test_size=0.2, random_state=seed)
# Convert data to np.float32
training_data = training_data.astype(np.float32)
validation_data = validation_data.astype(np.float32)
training_labels = training_labels.astype(np.float32)
validation_labels = validation_labels.astype(np.float32)

# print('training_data shape:', training_data.dtype)
# print('training_labels shape:', type(training_labels))
# print('validation_data shape:', type(validation_data))
# print('validation_labels shape:', type(validation_labels))

# =================================== LOAD DATA AND MODEL =========================================== #


def load_data(batch_size):
    train_dataset = AdultDataset(training_data, training_labels)
    valid_dataset = AdultDataset(validation_data, validation_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr, h_layer_size):
    model = MultiLayerPerceptron(np.shape(training_data)[1], h_layer_size)
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    for i, vbatch in enumerate(val_loader):
        features, label = vbatch

        # Run model on data
        prediction = model(features)

        # Check number of correct predictions
        corr = (prediction > 0.5).squeeze().long() == label.long()

        # Count number of correct predictions
        total_corr += int(corr.sum())

    return float(total_corr) / len(val_loader.dataset)


def main():
    batch_size = 64
    lr = 0.1
    epochs=20
    eval_every=10
    h_layer_size=50

    train_loader, val_loader = load_data(batch_size)
    model, loss_fnc, optimizer = load_model(lr, h_layer_size)

    t = 0
    training_accuracies = []
    validation_accuracies = []
    training_times = []
    gradient_steps = []

    prev_time = time()

    for epoch in range(epochs):
        accum_loss = 0
        total_correct = 0

        for i, batch in enumerate(train_loader):
            # Get one batch of data
            features, label = batch

            # Set all gradients to zero
            optimizer.zero_grad()

            # Run neural network on batch to get predictions
            predictions = model(features)

            # Compute loss function
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())

            accum_loss += batch_loss

            # Calculate gradients
            batch_loss.backward()

            # Update parameters
            optimizer.step()

            # Count number of correct predictions
            corr = (predictions > 0.5).squeeze().long() == label.long()
            total_correct += int(corr.sum())

            # Evaluate model every eval_every steps
            if (t + 1) % eval_every == 0:
                training_times.append(time() - prev_time)
                gradient_steps.append(t + 1)

                valid_acc = evaluate(model, val_loader)
                print("Epoch: {}, Step {} | Loss: {} | Valid acc: {}".format(epoch+1, t+1, accum_loss / eval_every,
                                                                             valid_acc))
                train_acc = evaluate(model, train_loader)
                training_accuracies.append(train_acc)
                validation_accuracies.append(valid_acc)
                accum_loss = 0

            t += 1

        # Output final accuracy
        print("Train acc:{}".format(float(total_correct) / len(train_loader.dataset)))

    # plt.plot(training_accuracies)
    # plt.plot(validation_accuracies)
    # plt.legend(['Training Accuracy', 'Validation Accuracy'])
    # plt.title('Training & Validation Accuracies vs. Gradient Step (lr = 0.01)')
    # plt.xlabel('Gradient Step')
    # plt.ylabel('Accuracy')
    # plt.show()

    window_length = 7
    polyorder = 3
    plt.plot(gradient_steps, signal.savgol_filter(training_accuracies, window_length, polyorder))
    plt.plot(gradient_steps, signal.savgol_filter(validation_accuracies, window_length, polyorder))
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('Training and Validation Accuracies vs. Gradient Step (Best)')
    plt.xlabel('Gradient Step')
    plt.ylabel('Accuracy')
    plt.savefig('fig', dpi=300)
    plt.show()

    np.save('training_accuracies_sigmoid.npy', training_accuracies)
    np.save('validation_accuracies_sigmoid.npy', validation_accuracies)
    np.save('gradients_steps_sigmoid.npy', gradient_steps)
    np.save('training_times_sigmoid.npy', training_times)

    # plt.plot(training_times, training_accuracies)
    # plt.plot(training_times, validation_accuracies)
    # plt.title('Training and Validation Accuracies vs. Time (batch_size = 1)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Accuracy')
    # plt.savefig('time', dpi=300)
    # plt.show()
