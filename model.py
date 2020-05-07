# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('data.csv')

dataset['RM'].fillna(dataset['RM'].median(), inplace=True)


X = dataset.iloc[:, :13]

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

#Fitting model with trainig data
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[-0.41604527,  0.31960611, -1.43969621, -0.27288841, -1.19699755,
        1.17504213, -0.32947699,  2.54755519, -0.75533878, -1.13684942,
        0.04879332,  0.38760272, -0.64069431]]))
