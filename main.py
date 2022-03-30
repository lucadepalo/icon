import pandas as pd
import numpy as np
import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

df = pd.read_csv('data/heart_2020_cleaned.csv')

print(df.head())
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
# This function returns the first n rows for the object based on position.
# parameter: n int, default 5. Number of rows to select.

# df.describe().T.style.set_properties(**{'background-color': 'grey', 'color': 'white', 'border-color': 'white'})
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html

df.info()
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html

print(df.nunique())
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html

df = df[df.columns].replace(
    {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': '0', 'Yes (during pregnancy)': '1'})
df['Diabetic'] = df['Diabetic'].astype(int)



fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["Sex"], bins=15, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["Sex"], bins=15, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("Sex")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No heart disease according to Sex")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))
# ax.set_xticks(2)
# ax.set_xticklabels(["Non fumatore","Fumatore"])
ax.hist(df[df["HeartDisease"] == 1]["Smoking"], bins=2, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["Smoking"], bins=2, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("Smoking")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No heart disease according to being a smoker or not.")

ax.legend()

plt.figure(figsize=(13, 6))
sns.countplot(x=df['Race'], hue='HeartDisease', data=df, palette='YlOrBr')
plt.xlabel('Race')
plt.ylabel('Frequency')
# plt.show()
print('Distribution of Cases with Yes/No heart disease according to being a smoker or not.')

plt.figure(figsize=(13, 6))
sns.countplot(x=df['AgeCategory'], hue='HeartDisease', data=df, palette='YlOrBr')
fig.suptitle("Distribution of Cases with Yes/No hartdisease according to AgeCategory")
plt.xlabel('AgeCategory')
plt.ylabel('Frequency')
# plt.show()
print('Distribution of Cases with Yes/No hartdisease according to AgeCategory')

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["KidneyDisease"], bins=15, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["KidneyDisease"], bins=15, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("KidneyDisease")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No heartdisease according to kidneydisease")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["SkinCancer"], bins=15, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["SkinCancer"], bins=15, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("SkinCancer")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No heartdisease based on previous exposure to skin cancer")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["Stroke"], bins=15, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["Stroke"], bins=15, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("Stroke")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No hartdisease based on previous exposure to Stroke")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["Diabetic"], bins=15, alpha=0.5, color="red", label="HeartDisease")
ax.hist(df[df["HeartDisease"] == 0]["Diabetic"], bins=15, alpha=0.5, color="#fccc79", label="Normal")

ax.set_xlabel("Diabetic")
ax.set_ylabel("Frequency")

fig.suptitle("Distribution of Cases with Yes/No hartdisease based on previous exposure to Diabetic")

ax.legend()

correlation = df.corr().round(2)
plt.figure(figsize=(14, 7))
sns.heatmap(correlation, annot=True, cmap='YlOrBr')

sns.set_style('white')
sns.set_palette('YlOrBr')
plt.figure(figsize=(13, 6))
plt.title('Distribution of correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
# plt.show()
print('Distribution of correlation of features')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["BMI"], alpha=0.5, shade=True, color="red", label="HeartDisease", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["BMI"], alpha=0.5, shade=True, color="#fccc79", label="Normal", ax=ax)
plt.title('Distribution of Body Mass Index', fontsize=18)
ax.set_xlabel("BodyMass")
ax.set_ylabel("Frequency")
ax.legend();
# plt.show()
print('Distribution of Body Mass Index')

#### We can see that people who weigh less than 40 kg are more likely to get heart disease!

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["SleepTime"], alpha=0.5, shade=True, color="red", label="HeartDisease", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["SleepTime"], alpha=0.5, shade=True, color="#fccc79", label="Normal", ax=ax)
plt.title('Distribution of SleepTime values', fontsize=18)
ax.set_xlabel("SleepTime")
ax.set_ylabel("Frequency")
ax.legend();
# plt.show()
print('Distribution of SleepTime values')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["PhysicalHealth"], alpha=0.5, shade=True, color="red", label="HeartDisease",
            ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["PhysicalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Normal",
            ax=ax)
plt.title('Distribution of PhysicalHealth state for the last 30 days',
          fontsize=18)  # Read the introduction to know what the scale of numerical features mean
ax.set_xlabel("PhysicalHealth")
ax.set_ylabel("Frequency")
ax.legend();
# plt.show()
print('Distribution of PhysicalHealth state for the last 30 days')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["MentalHealth"], alpha=0.5, shade=True, color="red", label="HeartDisease",
            ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["MentalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Normal", ax=ax)
plt.title('Distribution of MenalHealth state for the last 30 days', fontsize=18)
ax.set_xlabel("MentalHealth")
ax.set_ylabel("Frequency")
ax.legend();
# plt.show()
print('Distribution of MenalHealth state for the last 30 days')

from sklearn.preprocessing import StandardScaler

num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
Scaler = StandardScaler()
df[num_cols] = Scaler.fit_transform(df[num_cols])

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

# Encoding categorical features
categ = df[['AgeCategory', 'Race', 'GenHealth']]
encoded_categ = pd.DataFrame(enc.fit_transform(categ).toarray())

# Likning the encoed_cateh with the df
df = pd.concat([df, encoded_categ], axis=1)

# Dropping the categorical features
df = df.drop(columns=['AgeCategory', 'Race', 'GenHealth'], axis=1)

# Select Features
features = df.drop(columns=['HeartDisease'], axis=1)

# Select Target
target = df['HeartDisease']

# Set Training and Testing Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=True, test_size=.2, random_state=44)

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# Building a model using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# Evaluate Model
knn_eval = evaluate_model(knn, X_test, y_test)

# Print result
print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Area Under Curve:', knn_eval['auc'])
print('Confusion Matrix:\n', knn_eval['cm'])

from sklearn import tree

# Building Decision Tree model
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Evaluate Model
clf_eval = evaluate_model(clf, X_test, y_test)

# Print result
print('Accuracy:', clf_eval['acc'])
print('Precision:', clf_eval['prec'])
print('Recall:', clf_eval['rec'])
print('F1 Score:', clf_eval['f1'])
print('Cohens Kappa Score:', clf_eval['kappa'])
print('Area Under Curve:', clf_eval['auc'])
print('Confusion Matrix:\n', clf_eval['cm'])

# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
clf_score = [clf_eval['acc'], clf_eval['prec'], clf_eval['rec'], clf_eval['f1'], clf_eval['kappa']]
knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(clf_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, clf_score, width=barWidth, edgecolor='white', label='Decision Tree')
ax1.bar(r2, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(clf_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(clf_eval['fpr'], clf_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(clf_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()
