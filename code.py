import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# load dataset
df = pd.read_csv("Iris.csv")

# remove id
df = df.drop("Id", axis=1)

print(df.head())

# -----------------------------
# 1 DESCRIPTIVE STATISTICS
# -----------------------------

print("Summary Statistics")
print(df.describe())

# -----------------------------
# 2 DISTRIBUTION ANALYSIS
# -----------------------------

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

for col in features:
    
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# -----------------------------
# 3 SKEWNESS AND KURTOSIS
# -----------------------------

for col in features:
    
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    
    print(f"{col}")
    print("Skewness:", skew)
    print("Kurtosis:", kurt)
    print()

# -----------------------------
# 4 QQ PLOT (GAUSSIAN CHECK)
# -----------------------------

for col in features:
    
    plt.figure()
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f"QQ Plot of {col}")
    plt.show()

# -----------------------------
# 5 BOOTSTRAP CONFIDENCE INTERVAL
# -----------------------------

def bootstrap_ci(data, n_iterations=1000, ci=95):

    means = []

    for i in range(n_iterations):
        
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    alpha = 100 - ci
    
    lower = np.percentile(means, alpha/2)
    upper = np.percentile(means, 100 - alpha/2)

    return lower, upper


petal = df["PetalLengthCm"]

lower, upper = bootstrap_ci(petal)

print("Bootstrap 95% CI for Petal Length Mean")
print(lower, upper)

# -----------------------------
# 6 ANOVA TEST BETWEEN SPECIES
# -----------------------------

setosa = df[df['Species']=='Iris-setosa']['PetalLengthCm']
versicolor = df[df['Species']=='Iris-versicolor']['PetalLengthCm']
virginica = df[df['Species']=='Iris-virginica']['PetalLengthCm']

F_stat, p_value = stats.f_oneway(setosa, versicolor, virginica)

print("ANOVA Test Result")
print("F Statistic:", F_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# -----------------------------
# 7 BOXPLOT COMPARISON
# -----------------------------

plt.figure(figsize=(6,4))
sns.boxplot(x="Species", y="PetalLengthCm", data=df)
plt.title("Petal Length Comparison")
plt.show()
