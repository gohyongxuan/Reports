# -*- coding: utf-8 -*-
import sys
try:
    import bambi as bmb
except ImportError:
    !{sys.executable} -m pip install --upgrade bambi
    import bambi as bmb
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
import math
import arviz as az
import pytensor.tensor as at

df = pd.read_csv('resale_flat_prices.csv')
df.head()

def aff(x):
    if x >= 600000:
        return 0
    else:
        return 1
def get_region(row):
    town = row["town"]
    if town in ['ANG MO KIO','BISHAN', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 'KALLANG/WHAMPOA', 'TOA PAYOH']:
        return 6
    elif town in ['BUKIT MERAH', 'QUEENSTOWN', 'CLEMENTI']:
        return 5
    elif town in ['MARINE PARADE', 'BEDOK', 'PASIR RIS', 'TAMPINES']:
        return 4
    elif town in ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'JURONG EAST', 'JURONG WEST']:
        return 1
    elif town in ['HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON']:
        return 3
    elif town in ['SEMBAWANG', 'WOODLANDS', 'YISHUN']:
        return 2
    else:
        return None
df['year'] = df['month'].apply(lambda x: int(x.split('-')[0]))
df['year_sq'] = df['year'].apply(lambda x: x**2)
df['remaining_lease'] = df['remaining_lease'].apply(lambda x: int(x.split(' ')[0]))
df['storey_range'] = df['storey_range'].apply(lambda x: (int(x.split(' ')[0]) + int(x.split(' ')[2]))/2)
df['affordability'] = df['resale_price'].apply(aff)
df["region"] = df.apply(get_region, axis=1)
le = preprocessing.LabelEncoder()
df['flat_type'] = le.fit_transform(df['flat_type'])
df = df.drop(['block','street_name','lease_commence_date','town','flat_model','floor_area_sqm','month'],axis=1)
df.head(3)

df.describe()

numeric_features = df[['year', 'storey_range','remaining_lease']]

cols = numeric_features.columns

numeric_features.loc[:, cols] = preprocessing.scale(numeric_features.loc[:, cols])

data = pd.concat([numeric_features, df[['affordability', 'flat_type','region']]], axis=1)
data.head()

X = data.drop(columns=['affordability'])
y = data['affordability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test,y_test],axis=1)

func = 'affordability ~ region + year + flat_type + storey_range + remaining_lease'

model_2 = bmb.Model(func, train_data, family='bernoulli')
idata_2 = model_2.fit(chains=4)

az.plot_trace(idata_2)
plt.tight_layout()

pm.summary(idata_2)

intercept_samples = idata_2.posterior['Intercept'].values.flatten()
coef_samples_flat_type = idata_2.posterior['flat_type'].values.flatten()
coef_samples_remaining_lease = idata_2.posterior['remaining_lease'].values.flatten()
coef_samples_storey_range = idata_2.posterior['storey_range'].values.flatten()
coef_samples_year = idata_2.posterior['year'].values.flatten()
coef_samples_region = idata_2.posterior['region'].values.flatten()

def logistic(x):
    return 1 / (1 + np.exp(-x))

predictions = []

for i in range(len(intercept_samples)):
    linear_pred = (
        intercept_samples[i] +
        coef_samples_flat_type[i] * X_test['flat_type'] +
        coef_samples_remaining_lease[i] * X_test['remaining_lease'] +
        coef_samples_storey_range[i] * X_test['storey_range'] +
        coef_samples_year[i] * X_test['year'] +
        coef_samples_region[i] * X_test['region']
    )

    predicted_prob = logistic(linear_pred)

    simulated_outcomes = np.random.binomial(1, predicted_prob)
    predictions.append(simulated_outcomes)

ppc_results = pd.DataFrame(predictions).T

mean_ppc = ppc_results.mean(axis=1)
actuals = test_data['affordability'].values

plt.figure(figsize=(10, 6))
sns.histplot(actuals, kde=False, color="blue", label="Actual", bins=2, stat="density", alpha=0.5)

plt.xlabel("Affordability (0 = More than 600K, 1 = Less than 600K)")
plt.ylabel("Density")
plt.legend()
plt.title("Posterior Predictive Check: Actual vs Predicted Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(mean_ppc, kde=False, color="red", label="Predicted", bins=2, stat="density", alpha=0.5)

plt.xlabel("Affordability (0 = More than 600K, 1 = Less than 600K)")
plt.ylabel("Density")
plt.legend()
plt.title("Posterior Predictive Check: Actual vs Predicted Distribution")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(actuals, kde=False, color="blue", label="Actual", bins=2, stat="density", alpha=0.5, ax=axes[0])
axes[0].set_xlabel("Affordability (0 = More than 600K, 1 = Less than 600K)")
axes[0].set_ylabel("Density")
axes[0].set_title("Actual Outcomes Distribution")
axes[0].legend()

sns.histplot(mean_ppc, kde=False, color="red", label="Predicted", bins=2, stat="density", alpha=0.5, ax=axes[1])
axes[1].set_xlabel("Affordability (0 = More than 600K, 1 = Less than 600K)")
axes[1].set_ylabel("Density")
axes[1].set_title("Predicted Outcomes Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(actuals, kde=False, color="blue", label="Actual", bins=[-0.5, 0.5, 1.5], stat="density", alpha=0.5, discrete=True)
sns.histplot(mean_ppc, kde=False, color="red", label="Predicted", bins=[-0.5, 0.5, 1.5], stat="density", alpha=0.5, discrete=True)

plt.xlabel("Affordability (0 = Flat Price More than 600K, 1 = Flat Price Less than 600K)")
plt.ylabel("Density")
plt.title("Actual vs Predicted Outcomes Distribution")
plt.legend()
plt.xticks([0, 1])
plt.show()

logloss = log_loss(actuals, mean_ppc)
print("Log Loss:", logloss)
