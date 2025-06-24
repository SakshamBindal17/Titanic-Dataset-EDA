# Titanic Dataset - Exploratory Data Analysis (EDA)

[Download the Titanic Dataset](https://github.com/SakshamBindal17/Titanic-Dataset-Cleaning/blob/main/Titanic-Dataset-Cleaned.csv)

## Overview

This project presents a step-by-step Exploratory Data Analysis (EDA) on the cleaned Titanic dataset. The goal is to understand the data, visualize important features, identify patterns and anomalies, and make feature-level inferences that will guide future machine learning tasks.

---

## Steps Performed

### **1. Load and Summarize the Data**

We begin by loading the cleaned dataset and generating summary statistics.
```
import pandas as pd

df = pd.read_csv('Titanic-Dataset-Cleaned.csv')
print(df.describe())
print(df.describe(include='all'))
```

**Key Points:**
- No missing values remain in the dataset.
- `Sex` and `Embarked` columns are encoded numerically.
- `Age` and `Fare` are standardized (mean ≈ 0, std ≈ 1).

---

### **2. Visualize the Data**

#### **A. Histograms for Numerical Columns**
```
import matplotlib.pyplot as plt

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols].hist(figsize=(10, 8), bins=20)
plt.tight_layout()
plt.show()
```

#### **B. Boxplots for Numerical Columns**
```
import seaborn as sns

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
plt.subplot(2, 2, i)
sns.boxplot(x=df[col])
plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
```

#### **C. Countplots for Categorical Columns**

```
sns.countplot(x='Survived', data=df)
plt.title('Count of Survived')
plt.show()

sns.countplot(x='Sex', data=df)
plt.title('Count of Sex')
plt.show()

sns.countplot(x='Pclass', data=df)
plt.title('Count of Pclass')
plt.show()
```

---

### **3. Correlation Analysis**

#### **A. Correlation Matrix**

```
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

#### **B. Pairplot (Optional)**

```
sns.pairplot(df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']])
plt.show()
```

---

### **4. Patterns, Trends, and Anomalies**

- **Survived:** The dataset is imbalanced: about 62% did not survive, 38% survived.
- **Sex:** Females (Sex=1) had a much higher survival rate than males. This is confirmed by the positive correlation (0.54) between Sex and Survived.
- **Pclass:** Passengers in 1st class had a higher survival rate. Pclass is negatively correlated with Survived (-0.34).
- **Fare:** Passengers who paid higher fares were more likely to survive. Fare is positively correlated with Survived (0.26). There are a few extreme outliers (very high fares).
- **Age:** Most passengers were young adults. Age is not strongly correlated with survival, but some children survived.
- **SibSp & Parch:** Most passengers traveled alone. Having more siblings/spouses or parents/children does not strongly affect survival, but most survivors had 0 or 1 family member with them.
- **Embarked:** Most passengers embarked at port S. The embarkation port does not have a strong relationship with survival.
- **Anomalies:** Some passengers have very high Fare, SibSp, or Parch values (outliers). These cases are rare and may represent large families or wealthy individuals.

---

### **5. Feature-Level Inferences**

| Feature    | Inference |
|------------|-----------|
| **Sex**    | Being female greatly increased the chance of survival. |
| **Pclass** | Higher class (1st) passengers had better survival rates. |
| **Fare**   | Paying a higher fare was associated with higher survival. |
| **Age**    | Young adults were most common; children had some survival advantage. |
| **SibSp**  | Most passengers traveled alone; large families were rare. |
| **Parch**  | Most had no parents/children with them; large family groups were rare. |
| **Embarked** | Most embarked at 'S'; embarkation point had little effect on survival. |

#### **Embarked Encoding Table**

| Embarked_Q | Embarked_S | Original Embarked |
|------------|------------|-------------------|
| 0          | 0          | C                 |
| 0          | 1          | S                 |
| 1          | 0          | Q                 |

---

## **Conclusion**

This EDA reveals that **Sex**, **Pclass**, and **Fare** are the most important features for predicting survival on the Titanic. The dataset is imbalanced, and there are some outliers, but overall the data is well-prepared for machine learning tasks.

---

**Prepared by Saksham Bindal**
