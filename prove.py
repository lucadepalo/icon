import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# importa filtro warnings
from warnings import simplefilter

# ignora tutti i future warnings
simplefilter(action='ignore', category=FutureWarning)

# legge i dati dal csv
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
df = pd.read_csv('data/heart.csv')

# sostituisce stringhe con valori di verità
#df = df[df.columns].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': '0', 'Yes (during pregnancy)': '1'})
#df['Diabetic'] = df['Diabetic'].astype(int)

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#from sklearn.preprocessing import StandardScaler
#num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
#Scaler = StandardScaler()
#df[num_cols] = Scaler.fit_transform(df[num_cols])

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()

# Codifica delle features categoriche
#categ = df[['AgeCategory', 'Race', 'GenHealth']]
#encoded_categ = pd.DataFrame(enc.fit_transform(categ).toarray())

# Collegamento delle feature caregoriche codificate con il data frame
#df = pd.concat([df, encoded_categ], axis=1)

# Pulizia delle colonne
#df = df.drop(columns=['AgeCategory', 'Race', 'GenHealth'], axis=1)

# Selezione delle features
#features = df.drop(columns=['HeartDisease'], axis=1)

# Selezione del target
#target = df['HeartDisease']

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html
df.info()

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html
print(df.nunique())

# Metrics for Evaluation of model Accuracy and F1-score
from sklearn.metrics import f1_score, accuracy_score

# Importing the Decision Tree from scikit-learn library
from sklearn.tree import DecisionTreeClassifier

# For splitting of data into train and test set
from sklearn.model_selection import train_test_split

# first we split our data into input and output
# y is the output and is stored in "Class" column of dataframe
# X contains the other columns and are features or input
y = df.target
df.drop(['target'], axis=1, inplace=True)
X = df

# Now we split the dataset in train and test part
# here the train set is 75% and test set is 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# Training the model is as simple as this
# Use the function imported above and apply fit() on it
DT= DecisionTreeClassifier()
DT.fit(X_train,y_train)

# We use the predict() on the model to predict the output
pred = DT.predict(X_test)

# for classification we use accuracy and F1 score
print(accuracy_score(y_test, pred))
print(f1_score(y_test, pred))

# for regression we use R2 score and MAE(mean absolute error)
# all other steps will be same as classification as shown above
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print(mean_absolute_error(y_test, pred))
print(mean_absolute_error(y_test, pred))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)
DT2= DecisionTreeClassifier()
DT2.fit(X_train,y_train)
pred2 = DT2.predict(X_test)
print(accuracy_score(y_test, pred2))
print(f1_score(y_test, pred2))
print(mean_absolute_error(y_test, pred2))
print(mean_absolute_error(y_test, pred2))