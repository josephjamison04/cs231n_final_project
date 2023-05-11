import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_pickle("/home/ubuntu/CS231N/data/mini-data/small_images_all_data.pkl")
labels = pd.read_pickle("/home/ubuntu/CS231N/data/mini-data/small_images_labels.pkl")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=231)

X_test.to_pickle("test_data.pkl")
y_test.to_pickle("test_labels.pkl")

del X_test
del y_test
del data
del labels

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=231)

X_train.to_pickle("train_data.pkl")
y_train.to_pickle("train_labels.pkl")
X_valid.to_pickle("valid_data.pkl")
y_valid.to_pickle("valid_labels.pkl")
