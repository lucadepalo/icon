import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# legge i dati dal csv
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
df = pd.read_csv('data/heart_2020_cleaned.csv')

# sostituisce stringhe con valori di verità
df = df[df.columns].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': '0', 'Yes (during pregnancy)': '1'})
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
ax.hist(df[df["HeartDisease"] == 1]["Smoking"], bins=15, alpha=0.5, color="red", label="Cardiopatia")
ax.hist(df[df["HeartDisease"] == 0]["Smoking"], bins=15, alpha=0.5, color="#fccc79", label="Sano")
ax.set_xlabel("Fumatore")
ax.set_ylabel("Frequenza")
fig.suptitle("Distribuzione di casi positivi e negativi in relazione al tabagismo.")
ax.legend()

plt.figure(figsize=(13, 6))
sns.countplot(x=df['Race'], data=df)
plt.xlabel('Etnia')
plt.ylabel('Frequenza')
plt.show()
print("Distribuzione di casi positivi e negativi in relazione all'etnia")

plt.figure(figsize=(13, 6))
sns.countplot(x=df['AgeCategory'], data=df)  # hue='HeartDisease'
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
# sns.set_palette('YlOrBr')
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
sns.kdeplot(df[df["HeartDisease"] == 1]["PhysicalHealth"], alpha=0.5, shade=True, color="red", label="Cardiopatia", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["PhysicalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Sano", ax=ax)
plt.title('Distribuzione dello stato di salute fisica nell\'ultimo mese', fontsize=18)
ax.set_xlabel("Salute fisica")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione dello stato di salute fisica nell\'ultimo mese')

fig, ax = plt.subplots(figsize=(13, 5))
sns.kdeplot(df[df["HeartDisease"] == 1]["MentalHealth"], alpha=0.5, shade=True, color="red", label="Cardiopatia", ax=ax)
sns.kdeplot(df[df["HeartDisease"] == 0]["MentalHealth"], alpha=0.5, shade=True, color="#fccc79", label="Sano", ax=ax)
plt.title('Distribuzione dello stato di salute mentale nell\'ultimo mese', fontsize=18)
ax.set_xlabel("Salute mentale")
ax.set_ylabel("Frequenza")
ax.legend()
plt.show()
print('Distribuzione dello stato di salute mentale nell\'ultimo mese')
