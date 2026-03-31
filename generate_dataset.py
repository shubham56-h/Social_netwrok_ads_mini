"""
Step 1: Dataset Generation
Generates a synthetic Social Network Ads dataset with realistic relationships,
real-world issues (missing values, outliers, duplicates, inconsistent gender),
and derived-but-useless features (AgeGroup, SalarySlab).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# --- Core Feature Generation ---

# User IDs
user_ids = np.arange(10001, 10001 + N)

# Gender with inconsistent values (real-world noise)
# Weights must sum to 1.0 across exactly 6 options
gender_options = ['Male', 'male', 'M', 'Female', 'female', 'F']
gender_weights = [0.25, 0.10, 0.07, 0.25, 0.10, 0.23]  # sums to 1.0
gender_raw = np.random.choice(gender_options, size=N, p=gender_weights)

# Age: mostly 18-65, with some outliers
age = np.random.normal(loc=38, scale=12, size=N).astype(int)
# Introduce outliers: unrealistic ages
outlier_age_idx = np.random.choice(N, size=20, replace=False)
age[outlier_age_idx] = np.random.choice([5, 120, 150, 200], size=20)
age = np.clip(age, 1, 200)  # keep high outliers

# EstimatedSalary: mostly 20k-150k, with some very high outliers
salary = np.random.normal(loc=65000, scale=25000, size=N).astype(int)
outlier_sal_idx = np.random.choice(N, size=25, replace=False)
salary[outlier_sal_idx] = np.random.choice([500000, 750000, 1000000], size=25)
salary = np.clip(salary, 10000, 1100000)

# Add small noise to numeric columns
age = age + np.random.randint(-1, 2, size=N)
age = np.clip(age, 1, 200)
salary = salary + np.random.randint(-500, 501, size=N)

# Introduce negative ages AFTER noise — ensures exactly 30 negatives survive
negative_age_idx = np.random.choice(N, size=30, replace=False)
age[negative_age_idx] = np.random.choice([-1, -5, -12, -23, -7], size=30)

# Introduce negative salaries AFTER noise — ensures exactly 40 negatives survive
negative_sal_idx = np.random.choice(N, size=40, replace=False)
salary[negative_sal_idx] = np.random.choice([-500, -1200, -8000, -45000, -300], size=40)

# --- Target Variable: Purchased ---
# Probability influenced by Age and Salary (realistic relationship)
# Use percentile-based normalization so both classes are well represented
age_norm = (age - np.percentile(age, 10)) / (np.percentile(age, 90) - np.percentile(age, 10))
sal_norm = (salary - np.percentile(salary, 10)) / (np.percentile(salary, 90) - np.percentile(salary, 10))
age_norm = np.clip(age_norm, 0, 1)
sal_norm = np.clip(sal_norm, 0, 1)
purchase_prob = 0.35 * age_norm + 0.55 * sal_norm + 0.05 * np.random.rand(N)
purchase_prob = np.clip(purchase_prob, 0, 1)
# Use median as threshold to ensure ~50/50 split
threshold = np.median(purchase_prob)
purchased = (purchase_prob > threshold).astype(int)

# --- Derived but Useless Features ---
# AgeGroup: derived from Age (redundant with Age)
def assign_age_group(a):
    if a < 30:
        return 'Young'
    elif a < 50:
        return 'Adult'
    else:
        return 'Senior'

age_group = [assign_age_group(a) for a in age]

# SalarySlab: derived from EstimatedSalary (redundant with EstimatedSalary)
def assign_salary_slab(s):
    if s < 40000:
        return 'Low'
    elif s < 90000:
        return 'Medium'
    else:
        return 'High'

salary_slab = [assign_salary_slab(s) for s in salary]

# --- Build DataFrame ---
df = pd.DataFrame({
    'UserID': user_ids,
    'Gender': gender_raw,
    'Age': age.astype(float),
    'EstimatedSalary': salary.astype(float),
    'AgeGroup': age_group,
    'SalarySlab': salary_slab,
    'Purchased': purchased
})

# --- Introduce Missing Values ---
# Randomly set ~5% of Age and EstimatedSalary to NaN
missing_age_idx = np.random.choice(N, size=int(0.05 * N), replace=False)
missing_sal_idx = np.random.choice(N, size=int(0.05 * N), replace=False)
df.loc[missing_age_idx, 'Age'] = np.nan
df.loc[missing_sal_idx, 'EstimatedSalary'] = np.nan

# --- Introduce Duplicate Rows ---
duplicate_rows = df.sample(n=50, random_state=7)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# --- Save Dataset ---
df.to_csv('social_network_ads.csv', index=False)
print(f"Dataset saved: {len(df)} rows, {df.shape[1]} columns")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nGender unique values:", df['Gender'].unique())
