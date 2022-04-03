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
df = pd.read_csv('data/heart_2020_cleaned.csv')

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
# Questa funzione restituisce le prime n righe dell'oggetto basandosi sulla posizione.
# parametri: n int, default 5. Numero di righe selezionate.
print(df.head())

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
# df.describe().T.style.set_properties(**{'background-color': 'grey', 'color': 'white', 'border-color': 'white'})

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html
df.info()

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nunique.html
print(df.nunique())

# sostituisce stringhe con valori di verità
df = df[df.columns].replace(
    {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': '0', 'Yes (during pregnancy)': '1'})
df['Diabetic'] = df['Diabetic'].astype(int)

# mostra grafico
fig, ax = plt.subplots(figsize=(13, 6))

# imposta caratteristiche istogrammi
ax.hist(df[df["HeartDisease"] == 1]["Sex"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["Sex"], bins=15, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Sesso")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione al sesso")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))
ax.hist(df[df["HeartDisease"] == 1]["Smoking"], bins=2, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["Smoking"], bins=2, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Fumatore")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione al tabagismo.")

ax.legend()

plt.figure(figsize=(13, 6))
sns.countplot(x=df['Race'], data=df) # old: (x=df['Race'], hue='HeartDisease', data=df)
plt.xlabel('Etnia')
plt.ylabel('Frequenza')
plt.show()
print("Distribuzione di casi positivi e negativi in relazione all'etnia")

plt.figure(figsize=(13, 6))
sns.countplot(x=df['AgeCategory'], data=df) # hue='HeartDisease'
fig.suptitle("Distribuzione di casi positivi e negativi in relazione alla fascia d'età")
plt.xlabel('Fascia d\'età')
plt.ylabel('Frequenza')
plt.show()
print('Distribuzione di casi positivi e negativi in relazione alla fascia d\'età')

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["KidneyDisease"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["KidneyDisease"], bins=15, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Nefropatia")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione alle nefropatie")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["SkinCancer"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["SkinCancer"], bins=15, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Cancro alla pelle")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione al cancro alla pelle")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["Stroke"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["Stroke"], bins=15, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Ictus")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione a ictus")

ax.legend()

fig, ax = plt.subplots(figsize=(13, 6))

ax.hist(df[df["HeartDisease"] == 1]["Diabetic"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["Diabetic"], bins=15, alpha=0.5, color="#fccc79", label="Sano")

ax.set_xlabel("Diabetico")
ax.set_ylabel("Frequenza")

fig.suptitle("Distribuzione di casi positivi e negativi in relazione al diabete")

ax.legend()

correlation = df.corr().round(2)
plt.figure(figsize=(14, 7))
sns.heatmap(correlation, annot=True, cmap='YlOrBr')

sns.set_style('white')
#sns.set_palette('YlOrBr')
plt.figure(figsize=(13, 6))
plt.title('Distribuzione della correlazione di features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()
print('Distribuzione della correlazione di features')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["BMI"], alpha=0.5, shade=True, color="red", label="Cardiopatia", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["BMI"], alpha=0.5, shade=True, color="#fccc79", label="Sano", ax=ax)
plt.title('Distribuzione dell\'indice di massa corporea', fontsize=18)
ax.set_xlabel("Massa corporea")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione dell\'indice di massa corporea')

#### We can see that people who weigh less than 40 kg are more likely to get heart disease!

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["SleepTime"], alpha=0.5, shade=True, color="red", label="Cardiopatia", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["SleepTime"], alpha=0.5, shade=True, color="#fccc79", label="Sano", ax=ax)
plt.title('Distribuzione delle ore di sonno', fontsize=18)
ax.set_xlabel("Ore di sonno")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione delle ore di sonno')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["PhysicalHealth"], alpha=0.5, shade=True, color="red", label="Cardiopatia",
            ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["PhysicalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Sano",
            ax=ax)
plt.title('Distribuzione dello stato di salute fisica nell\'ultimo mese',
          fontsize=18)  # Read the introduction to know what the scale of numerical features mean
ax.set_xlabel("Salute fisica")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione dello stato di salute fisica nell\'ultimo mese')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["MentalHealth"], alpha=0.5, shade=True, color="red", label="Cardiopatia",
            ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["MentalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Sano", ax=ax)
plt.title('Distribuzione dello stato di salute mentale nell\'ultimo mese', fontsize=18)
ax.set_xlabel("Salute mentale")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione dello stato di salute fisica nell\'ultimo mese')

from sklearn.preprocessing import StandardScaler
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
Scaler = StandardScaler()
df[num_cols] = Scaler.fit_transform(df[num_cols])

from sklearn.preprocessing import OneHotEncoder
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

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
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=True, test_size=.2, random_state=44)

print('Dimensioni del training feature:', X_train.shape)
print('Dimensioni del testing feature:', X_test.shape)
print('Dimensioni del training label:', y_train.shape)
print('Dimensioni del training label:', y_test.shape)


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
    # cm = metrics.confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) #nuovo

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'y_pred': y_pred }

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(max_iter=150)
logreg.fit(X_train,y_train)
logreg_eval=evaluate_model(logreg, X_test, y_test)
print('\n Regressione Logistica')
print('Accuracy:', logreg_eval['acc'])
print('Precision:', logreg_eval['prec'])
print('Recall:', logreg_eval['rec'])
print('F1 Score:', logreg_eval['f1'])
print('Cohens Kappa Score:', logreg_eval['kappa'])
print('Area Under Curve:', logreg_eval['auc'])
disp = ConfusionMatrixDisplay(confusion_matrix=logreg_eval['cm']) #nuovo
disp.plot()
plt.show()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_eval = evaluate_model(gnb, X_test, y_test)
print('\n Naive Bayes gaussiano ')
print('Accuracy:', gnb_eval['acc'])
print('Precision:', gnb_eval['prec'])
print('Recall:', gnb_eval['rec'])
print('F1 Score:', gnb_eval['f1'])
print('Cohens Kappa Score:', gnb_eval['kappa'])
print('Area Under Curve:', gnb_eval['auc'])
disp = ConfusionMatrixDisplay(confusion_matrix=gnb_eval['cm']) #nuovo
disp.plot()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_eval = evaluate_model(knn, X_test, y_test)
print('\n classificatore K nearest neighbors')
print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Area Under Curve:', knn_eval['auc'])
disp = ConfusionMatrixDisplay(confusion_matrix=knn_eval['cm']) #nuovo
disp.plot()
plt.show()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_eval = evaluate_model(rfc, X_test, y_test)
print('\n classificatore random forest')
print('Accuracy:', rfc_eval['acc'])
print('Precision:', rfc_eval['prec'])
print('Recall:', rfc_eval['rec'])
print('F1 Score:', rfc_eval['f1'])
print('Cohens Kappa Score:', rfc_eval['kappa'])
print('Area Under Curve:', rfc_eval['auc'])
disp = ConfusionMatrixDisplay(confusion_matrix=rfc_eval['cm']) #nuovo
disp.plot()
plt.show()

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=100)
abc.fit(X_train, y_train)
abc_eval = evaluate_model(abc, X_test, y_test)
print('\n classificatore Ada Boost')
print('Accuracy:', abc_eval['acc'])
print('Precision:', abc_eval['prec'])
print('Recall:', abc_eval['rec'])
print('F1 Score:', abc_eval['f1'])
print('Cohens Kappa Score:', abc_eval['kappa'])
print('Area Under Curve:', abc_eval['auc'])
disp = ConfusionMatrixDisplay(confusion_matrix=abc_eval['cm']) #nuovo
disp.plot()
plt.show()

# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Confronto dei modelli', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]
logreg_score = [logreg_eval['acc'], logreg_eval['prec'], logreg_eval['rec'], logreg_eval['f1'], logreg_eval['kappa']]
rfc_score = [rfc_eval['acc'], rfc_eval['prec'], rfc_eval['rec'], rfc_eval['f1'], rfc_eval['kappa']]
gnb_score = [gnb_eval['acc'], gnb_eval['prec'], gnb_eval['rec'], gnb_eval['f1'], gnb_eval['kappa']]
abc_score = [abc_eval['acc'], abc_eval['prec'], abc_eval['rec'], abc_eval['f1'], abc_eval['kappa']]


## Set position of bar on X axis
r1 = np.arange(len(knn_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]



## Make the plot
ax1.bar(r1, knn_score, width=barWidth, edgecolor='white', label='K Nearest Neighbors')
ax1.bar(r2, logreg_score, width=barWidth, edgecolor='white', label='Regressione Logistica')
ax1.bar(r3, gnb_score, width=barWidth, edgecolor='white', label='Naive Bayes Gaussiano')
ax1.bar(r4, rfc_score, width=barWidth, edgecolor='white', label='Foresta Casuale')
ax1.bar(r5, abc_score, width=barWidth, edgecolor='white', label='Ada Boost')

## Configure x and y axis
ax1.set_xlabel('Metriche', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(knn_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Punteggio', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Metriche di valutazione', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='KNN, auc = {:0.5f}'.format(knn_eval['auc']))
ax2.plot(logreg_eval['fpr'], logreg_eval['tpr'], label='Regressione Logistica, auc = {:0.5f}'.format(logreg_eval['auc']))
ax2.plot(gnb_eval['fpr'], gnb_eval['tpr'], label='Naive Bayes Gaussiano, auc = {:0.5f}'.format(gnb_eval['auc']))
ax2.plot(rfc_eval['fpr'], rfc_eval['tpr'], label='Foresta Casuale, auc = {:0.5f}'.format(rfc_eval['auc']))
ax2.plot(abc_eval['fpr'], abc_eval['tpr'], label='Ada Boost, auc = {:0.5f}'.format(abc_eval['auc']))


## Configure x and y axis
ax2.set_xlabel('Tasso di falsi positivi', fontweight='bold')
ax2.set_ylabel('Tasso di veri positivi', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()
