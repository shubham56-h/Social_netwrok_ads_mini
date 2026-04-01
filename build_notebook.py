import json

def md(id_, src):
    return {'cell_type': 'markdown', 'id': id_, 'metadata': {}, 'source': src}

def code(id_, src):
    return {'cell_type': 'code', 'execution_count': None, 'id': id_, 'metadata': {}, 'outputs': [], 'source': src}

# ─────────────────────────────────────────────────────────────────────────────
# TITLE & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
title = md('a1', [
    '# Social Network Ads — ML Pipeline\n',
    '## EDA-1 → Preprocessing → EDA-2 → Hyperparameter Tuning → Model Saving\n',
    '\n',
    '**Flow:** Explore raw data → discover issues → clean based on insights → confirm with EDA-2 → train models'
])

imports = code('a2', [
    'import pandas as pd\n',
    'import numpy as np\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    'from sklearn.model_selection import train_test_split, GridSearchCV\n',
    'from sklearn.preprocessing import StandardScaler\n',
    'from sklearn.linear_model import LogisticRegression\n',
    'from sklearn.neighbors import KNeighborsClassifier\n',
    'from sklearn.svm import SVC\n',
    'from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score\n',
    'import joblib, os\n',
    '%matplotlib inline\n',
    "os.makedirs('static', exist_ok=True)"
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
load_md   = md('b1', ['---\n', '## Step 2: Load Dataset'])
load_code = code('b2', [
    "df = pd.read_csv('social_network_ads.csv')\n",
    "print(f'Shape: {df.shape}')\n",
    'df.head()'
])
dtypes_code = code('b3', [
    "print('Data Types:')\n",
    'print(df.dtypes)\n',
    "print('\\nMissing Values:')\n",
    'print(df.isnull().sum())'
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EDA PART 1 (on raw data)
# ─────────────────────────────────────────────────────────────────────────────
eda1_md = md('c1', [
    '---\n',
    '## Step 3: EDA — Part 1 (Raw Data)\n',
    '\n',
    'Explore the **raw, uncleaned** dataset to discover issues before fixing them.\n',
    'Insights from here will directly guide our preprocessing decisions.'
])

eda1_stats = code('c2', [
    "print('=== Summary Statistics ===')\n",
    'df.describe()'
])

eda1_missing = code('c3', [
    "print('=== Missing Value Counts ===')\n",
    'print(df.isnull().sum())\n',
    '\n',
    '# Percentage bar plot — shows which columns have missing data and how much\n',
    'missing_pct = df.isnull().mean() * 100\n',
    'missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)\n',
    'plt.figure(figsize=(6, 3))\n',
    "bars = plt.bar(missing_pct.index, missing_pct.values, color='salmon', edgecolor='black')\n",
    'for bar, val in zip(bars, missing_pct.values):\n',
    '    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,\n',
    '             f\'{val:.1f}%\', ha=\'center\', va=\'bottom\', fontsize=10)\n',
    "plt.title('Missing Value Percentage by Column')\n",
    "plt.ylabel('Missing %')\n",
    'plt.ylim(0, missing_pct.max() + 5)\n',
    "plt.tight_layout(); plt.savefig('static/eda1_missing_pct.png'); plt.show()\n",
    '\n',
    '# Heatmap — only Age & Salary (the columns that actually have missing values)\n',
    'plt.figure(figsize=(6, 3))\n',
    "sns.heatmap(df[['Age', 'EstimatedSalary']].isnull(), cbar=False, cmap='viridis', yticklabels=False)\n",
    "plt.title('Missing Value Heatmap — Age & Salary (yellow = missing)')\n",
    "plt.tight_layout(); plt.savefig('static/eda1_missing_heatmap.png'); plt.show()"
])

eda1_gender = code('c4', [
    "print('=== Gender Unique Values ===')\n",
    'print(df["Gender"].value_counts())\n',
    '\n',
    "print('\\n=== AgeGroup Unique Values ===')\n",
    'print(df["AgeGroup"].value_counts())\n',
    '\n',
    "print('\\n=== SalarySlab Unique Values ===')\n",
    'print(df["SalarySlab"].value_counts())'
])

eda1_gender_corr = code('c4b', [
    '# Crosstab heatmap: Gender vs Purchased\n',
    '# Temporarily standardize Gender variants for display only (df is not modified)\n',
    "gender_map = {'Male': 'Male', 'male': 'Male', 'M': 'Male',\n",
    "              'Female': 'Female', 'female': 'Female', 'F': 'Female'}\n",
    "gender_std = df['Gender'].map(gender_map)\n",
    "ct = pd.crosstab(gender_std, df['Purchased'], normalize='index') * 100\n",
    "ct.columns = ['Not Purchased (0)', 'Purchased (1)']\n",
    'plt.figure(figsize=(5, 3))\n',
    "sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues',\n",
    "            linewidths=0.5, cbar_kws={'label': '% within Gender'})\n",
    "plt.title('Gender vs Purchased — Purchase Rate (%)')\n",
    "plt.ylabel('Gender'); plt.xlabel('')\n",
    "plt.tight_layout(); plt.savefig('static/eda1_gender_corr.png'); plt.show()\n",
    "print('Insight: if Male % ≈ Female % → Gender has little predictive power')"
])

eda1_neg = code('c5', [
    "print(f'Negative Ages   : {(df[\"Age\"] < 0).sum()}')\n",
    "print(f'Negative Salaries: {(df[\"EstimatedSalary\"] < 0).sum()}')\n",
    "print(f'Duplicate rows  : {df.duplicated().sum()}')\n",
    '\n',
    '# Show negative age rows\n',
    "print('\\nSample negative ages:')\n",
    "print(df[df['Age'] < 0][['UserID', 'Age']].head(8))\n",
    "print('\\nSample negative salaries:')\n",
    "print(df[df['EstimatedSalary'] < 0][['UserID', 'EstimatedSalary']].head(8))"
])

eda1_dist_age = code('c6a', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.histplot(df['Age'].dropna(), kde=True, color='steelblue', bins=40)\n",
    "plt.title('Raw Age Distribution (incl. negatives & outliers)')\n",
    "plt.tight_layout(); plt.savefig('static/eda1_age_dist.png'); plt.show()"
])

eda1_dist_sal = code('c6b', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.histplot(df['EstimatedSalary'].dropna(), kde=True, color='coral', bins=40)\n",
    "plt.title('Raw Salary Distribution (incl. negatives & outliers)')\n",
    '# Show actual salary values instead of scientific notation\n',
    'plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f\'{int(x):,}\'))\n',
    'plt.xticks(rotation=30, ha=\'right\')\n',
    "plt.tight_layout(); plt.savefig('static/eda1_salary_dist.png'); plt.show()"
])

eda1_box_age = code('c7a', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(y=df['Age'].dropna(), color='lightblue',\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),\n",
    "            showmeans=True)\n",
    "plt.title('Raw Age Boxplot')\n",
    "plt.tight_layout(); plt.savefig('static/eda1_age_box.png'); plt.show()"
])

eda1_box_sal = code('c7b', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(y=df['EstimatedSalary'].dropna(), color='lightyellow',\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),\n",
    "            showmeans=True)\n",
    "plt.title('Raw Salary Boxplot')\n",
    'plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f\'{int(x):,}\'))\n',
    "plt.tight_layout(); plt.savefig('static/eda1_salary_box.png'); plt.show()"
])

eda1_target = code('c8', [
    '# Class balance check\n',
    "print('Purchased value counts:')\n",
    "print(df['Purchased'].value_counts())\n",
    'plt.figure(figsize=(4, 3))\n',
    "sns.countplot(x='Purchased', data=df, palette='pastel')\n",
    "plt.title('Target Class Distribution')\n",
    "plt.tight_layout(); plt.savefig('static/eda1_target.png'); plt.show()"
])

eda1_insights = md('c9', [
    '### EDA-1 Insights → Preprocessing Actions\n',
    '\n',
    '| Insight Found | Action to Take |\n',
    '|---|---|\n',
    '| Duplicate rows present | Remove duplicates |\n',
    '| Gender has 6 inconsistent variants | Standardize to Male/Female |\n',
    '| Negative Age values | Replace with NaN → median impute |\n',
    '| Negative Salary values | Replace with NaN → median impute |\n',
    '| Missing values in Age & Salary | Median imputation |\n',
    '| Extreme outliers in Age & Salary (boxplots) | IQR-based removal |\n',
    '| AgeGroup & SalarySlab are derived columns | Drop — confirmed after correlation heatmap |\n',
    '| UserID is an identifier | Drop — no predictive signal |\n',
    '| Gender correlation with Purchased unknown yet | Encode Gender → run heatmap → decide |'
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PREPROCESSING (based on EDA-1 insights)
# ─────────────────────────────────────────────────────────────────────────────
prep_md = md('d1', [
    '---\n',
    '## Step 4: Data Preprocessing & Cleaning\n',
    '\n',
    'Each step below is directly motivated by an insight from EDA-1.'
])

# 4a duplicates
dup_md   = md('d2', ['### 4a. Remove Duplicate Rows\n', '**Why:** EDA-1 showed duplicate rows exist — they bias model training.'])
dup_code = code('d3', [
    'before = len(df)\n',
    'df.drop_duplicates(inplace=True)\n',
    "print(f'Duplicates removed: {before - len(df)} rows. Shape now: {df.shape}')"
])

# 4b gender
gender_md   = md('d4', ['### 4b. Standardize Gender Values\n', '**Why:** EDA-1 showed 6 inconsistent variants (Male, male, M, Female, female, F).'])
gender_code = code('d5', [
    "gender_map = {'Male': 'Male', 'male': 'Male', 'M': 'Male',\n",
    "              'Female': 'Female', 'female': 'Female', 'F': 'Female'}\n",
    "df['Gender'] = df['Gender'].map(gender_map)\n",
    "print('Gender after standardization:', df['Gender'].unique())\n",
    'print(df["Gender"].value_counts())'
])

# 4c negative age
negage_md   = md('d6', [
    '### 4c. Handle Invalid (Negative) Ages\n',
    '**Why:** EDA-1 revealed negative age values — logically impossible, caused by data entry errors.\n',
    'Replaced with `NaN` (not dropped) to preserve the rest of the row.'
])
negage_code = code('d7', [
    "neg_count = (df['Age'] < 0).sum()\n",
    "print(f'Negative age values found: {neg_count}')\n",
    "df['Age'] = df['Age'].where(df['Age'] >= 0, other=np.nan)\n",
    'print(f\'Missing Age count after replacement: {df["Age"].isna().sum()}\')'
])

# 4d negative salary
negsal_md   = md('d8', [
    '### 4d. Handle Invalid (Negative) Salaries\n',
    '**Why:** EDA-1 revealed negative salary values — a person cannot earn a negative income.\n',
    'Same approach as negative ages — replace with `NaN`.'
])
negsal_code = code('d9', [
    "neg_sal = (df['EstimatedSalary'] < 0).sum()\n",
    "print(f'Negative salary values found: {neg_sal}')\n",
    "df['EstimatedSalary'] = df['EstimatedSalary'].where(df['EstimatedSalary'] >= 0, other=np.nan)\n",
    'print(f\'Missing Salary count after replacement: {df["EstimatedSalary"].isna().sum()}\')'
])

# 4e imputation
impute_md   = md('d10', [
    '### 4e. Handle Missing Values — Median Imputation\n',
    '**Why:** EDA-1 showed missing values in Age & Salary. Median is used (not mean) because EDA-1 showed skewed distributions with outliers.\n',
    'Also fills the NaNs created from negative age/salary replacements above.\n',
    'After imputation, both `Age` and `EstimatedSalary` are cast to `int` — whole numbers only.'
])
impute_code = code('d11', [
    "age_median = df['Age'].median()\n",
    "df['Age'] = df['Age'].fillna(age_median)\n",
    "salary_median = df['EstimatedSalary'].median()\n",
    "df['EstimatedSalary'] = df['EstimatedSalary'].fillna(salary_median)\n",
    "df['Age'] = df['Age'].astype(int)\n",
    "df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)\n",
    "print('Missing values after imputation:')\n",
    'print(df.isnull().sum())\n',
    'print(f\'Age dtype: {df["Age"].dtype}\')'
])

# 4f drop columns — now handled in 4j above, remove old variables
# 4g outliers — now handled in 4k above, remove old variables

# 4h encode
encode_md   = md('d16', ['### 4h. Encode Gender — Male=1, Female=0\n', '**Why:** ML models require numeric input. Encoding is needed before we can run the correlation heatmap.'])
encode_code = code('d17', [
    "df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})\n",
    "print('Gender encoded: Male=1, Female=0')\n",
    'df.head()'
])

# 4i correlation heatmap → drop decision
corr_md = md('d18', [
    '### 4i. Correlation Heatmap → Feature Drop Decision\n',
    '\n',
    'Now that Gender is numeric, we can run a full correlation heatmap.\n',
    'This tells us which features actually relate to `Purchased` and which are noise.\n',
    '\n',
    '**Expected findings:**\n',
    '- `EstimatedSalary` → strong positive correlation with Purchased\n',
    '- `Age` → moderate positive correlation\n',
    '- `Gender` → near-zero correlation → **drop it**'
])
corr_code = code('d19', [
    '# Show only Gender vs Purchased correlation — Gender is now numeric (0/1)\n',
    "corr_val = df[['Gender', 'Purchased']].corr()\n",
    'plt.figure(figsize=(4, 3))\n',
    "sns.heatmap(corr_val, annot=True, cmap='coolwarm', fmt='.2f',\n",
    "            linewidths=0.5, square=True, vmin=-1, vmax=1)\n",
    "plt.title('Correlation: Gender vs Purchased')\n",
    "plt.tight_layout(); plt.savefig('static/corr_heatmap.png'); plt.show()\n",
    "print(f'Gender-Purchased correlation: {df[\"Gender\"].corr(df[\"Purchased\"]):.4f}')\n",
    "print('Insight: correlation ≈ 0 → Gender has no predictive power → will be dropped')"
])

# 4j drop columns (now includes Gender based on heatmap)
drop_md = md('d20', [
    '### 4j. Drop Useless Columns\n',
    '\n',
    'Based on the correlation heatmap and domain knowledge, we drop 4 columns:\n',
    '\n',
    '| Column | Reason |\n',
    '|---|---|\n',
    '| `Gender` | Correlation with Purchased ≈ -0.02 — near zero, adds noise not signal |\n',
    '| `UserID` | Identifier only — no predictive value |\n',
    '| `AgeGroup` | Derived from `Age` — redundant, causes multicollinearity |\n',
    '| `SalarySlab` | Derived from `EstimatedSalary` — redundant, causes multicollinearity |\n',
    '\n',
    '> Dropped **before** outlier treatment so IQR only runs on relevant columns.'
])
drop_code = code('d21', [
    "df.drop(columns=['Gender', 'UserID', 'AgeGroup', 'SalarySlab'], inplace=True)\n",
    "print('Remaining columns:', df.columns.tolist())"
])

# 4k outliers
outlier_md   = md('d22', [
    '### 4k. Outlier Treatment — IQR Method\n',
    '**Why:** EDA-1 boxplots clearly showed extreme outliers in Age and Salary.\n',
    'IQR method: remove values below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`.'
])
outlier_code = code('d23', [
    'def remove_outliers_iqr(dataframe, column):\n',
    '    Q1 = dataframe[column].quantile(0.25)\n',
    '    Q3 = dataframe[column].quantile(0.75)\n',
    '    IQR = Q3 - Q1\n',
    '    before = len(dataframe)\n',
    '    dataframe = dataframe[(dataframe[column] >= Q1 - 1.5*IQR) & (dataframe[column] <= Q3 + 1.5*IQR)]\n',
    '    print(f"  {column}: {before - len(dataframe)} outliers removed")\n',
    '    return dataframe\n',
    '\n',
    "df = remove_outliers_iqr(df, 'Age')\n",
    "df = remove_outliers_iqr(df, 'EstimatedSalary')\n",
    "print(f'Shape after outlier removal: {df.shape}')"
])

# 4l split & scale
split_md   = md('d24', [
    '### 4l. Train-Test Split & Feature Scaling\n',
    '**Why scaling:** Age (~18-65) and Salary (~20k-150k) are on very different scales.\n',
    'StandardScaler normalizes them so no feature dominates distance-based models (KNN).\n',
    '**Important:** Scaler is fit only on training data to prevent data leakage from test set.'
])
split_code = code('d25', [
    "X = df[['Age', 'EstimatedSalary']]\n",
    "y = df['Purchased']\n",
    'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n',
    "print(f'Train: {X_train.shape}  Test: {X_test.shape}')\n",
    'scaler = StandardScaler()\n',
    'X_train_scaled = scaler.fit_transform(X_train)\n',
    'X_test_scaled  = scaler.transform(X_test)\n',
    "print('Scaled with StandardScaler.')"
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EDA PART 2 (on clean data)
# ─────────────────────────────────────────────────────────────────────────────
eda2_md = md('e1', [
    '---\n',
    '## Step 5: EDA — Part 2 (Clean Data)\n',
    '\n',
    'Confirm preprocessing worked and extract meaningful insights for model building.'
])

eda2_stats = code('e2', [
    "print('=== Summary Statistics (Clean Data) ===')\n",
    'df.describe()'
])

eda2_dist_age = code('e3a', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.histplot(df['Age'], kde=True, color='steelblue', bins=30)\n",
    "plt.title('Age Distribution (Clean)')\n",
    "plt.tight_layout(); plt.savefig('static/eda2_age_dist.png'); plt.show()"
])

eda2_dist_sal = code('e3b', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.histplot(df['EstimatedSalary'], kde=True, color='coral', bins=30)\n",
    "plt.title('Salary Distribution (Clean)')\n",
    'plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f\'{int(x):,}\'))\n',
    'plt.xticks(rotation=30, ha=\'right\')\n',
    "plt.tight_layout(); plt.savefig('static/eda2_salary_dist.png'); plt.show()"
])

eda2_box_age = code('e4a', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(y=df['Age'], color='lightblue',\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),\n",
    "            showmeans=True)\n",
    "plt.title('Age Boxplot (Clean)')\n",
    "plt.tight_layout(); plt.savefig('static/eda2_age_box.png'); plt.show()"
])

eda2_box_sal = code('e4b', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(y=df['EstimatedSalary'], color='lightyellow',\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),\n",
    "            showmeans=True)\n",
    "plt.title('Salary Boxplot (Clean)')\n",
    'plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f\'{int(x):,}\'))\n',
    "plt.tight_layout(); plt.savefig('static/eda2_salary_box.png'); plt.show()"
])

eda2_corr = code('e5', [
    '# Correlation heatmap — Age & Salary vs Purchased on clean data\n',
    "corr_data = df[['Age', 'EstimatedSalary', 'Purchased']].corr()\n",
    'plt.figure(figsize=(5, 4))\n',
    "sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f',\n",
    "            linewidths=0.5, square=True)\n",
    "plt.title('Correlation Heatmap — Age & Salary vs Purchased')\n",
    "plt.tight_layout(); plt.savefig('static/eda2_corr_heatmap.png'); plt.show()\n",
    "print('Final feature columns:', df.columns.tolist())"
])

eda2_vs_age = code('e6a', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(x='Purchased', y='Age', data=df, hue='Purchased', palette='Set2', legend=False,\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6), showmeans=True)\n",
    "plt.title('Age vs Purchased')\n",
    "plt.tight_layout(); plt.savefig('static/eda2_age_vs_target.png'); plt.show()"
])

eda2_vs_sal = code('e6b', [
    'plt.figure(figsize=(6, 4))\n',
    "sns.boxplot(x='Purchased', y='EstimatedSalary', data=df, hue='Purchased', palette='Set3', legend=False,\n",
    "            meanprops=dict(marker='D', markerfacecolor='red', markersize=6), showmeans=True)\n",
    "plt.title('Salary vs Purchased')\n",
    'plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f\'{int(x):,}\'))\n',
    "plt.tight_layout(); plt.savefig('static/eda2_salary_vs_target.png'); plt.show()"
])

eda2_insights = md('e8', [
    '### EDA-2 Insights → Model Building Decisions\n',
    '\n',
    '| Insight | Impact on Model |\n',
    '|---|---|\n',
    '| Higher salary → more likely to purchase | Salary is the strongest feature |\n',
    '| Older age → slightly more likely to purchase | Age is a useful feature |\n',
    '| Gender was dropped (correlation ≈ -0.02) | Only Age & Salary used as features |\n',
    '| Age & Salary on different scales | Confirms StandardScaler was necessary |\n',
    '| Classes are roughly balanced | No need for class weighting or resampling |'
])

# Scatter plot — Age vs Salary colored by Purchased
eda2_scatter = code('e9', [
    '# Most powerful EDA chart — shows the decision boundary visually\n',
    'plt.figure(figsize=(7, 5))\n',
    "colors = df['Purchased'].map({0: '#E74C3C', 1: '#2ECC71'})\n",
    "plt.scatter(df['Age'], df['EstimatedSalary'], c=colors, alpha=0.5, edgecolors='none', s=20)\n",
    "from matplotlib.patches import Patch\n",
    "legend_elements = [Patch(facecolor='#E74C3C', label='Not Purchased (0)'),\n",
    "                   Patch(facecolor='#2ECC71', label='Purchased (1)')]\n",
    "plt.legend(handles=legend_elements)\n",
    "plt.xlabel('Age'); plt.ylabel('Estimated Salary')\n",
    "plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))\n",
    "plt.title('Age vs Salary — colored by Purchased')\n",
    "plt.tight_layout(); plt.savefig('static/eda2_scatter.png'); plt.show()\n",
    "print('Insight: high salary + older age → purchased (green cluster top-right)')"
])

# Class balance recheck after preprocessing
eda2_balance = code('e10', [
    "print('Class balance after preprocessing:')\n",
    "print(df['Purchased'].value_counts())\n",
    "print(f'\\nClass ratio: {df[\"Purchased\"].value_counts(normalize=True).round(3).to_dict()}')\n",
    'plt.figure(figsize=(4, 3))\n',
    "sns.countplot(x='Purchased', data=df, palette='pastel')\n",
    "plt.title('Class Balance After Preprocessing')\n",
    "plt.tight_layout(); plt.show()"
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────
model_md = md('f1', [
    '---\n',
    '## Step 6: Model Building, Hyperparameter Tuning & Evaluation\n',
    '\n',
    'GridSearchCV with 5-fold cross-validation.\n',
    '**Why these models:**\n',
    '- Logistic Regression — interpretable baseline; works well when boundary is roughly linear\n',
    '- KNN — non-parametric; captures non-linear patterns; sensitive to scale (hence StandardScaler)\n',
    '- SVM — finds the optimal hyperplane with maximum margin; works well in high-dimensional space; kernel trick handles non-linearity'
])

eval_fn = code('f2', [
    'def tune_and_evaluate(name, estimator, param_grid, X_tr, y_tr, X_te, y_te):\n',
    '    grid = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1)\n',
    '    grid.fit(X_tr, y_tr)\n',
    '    best = grid.best_estimator_\n',
    '    y_pred = best.predict(X_te)\n',
    '    y_prob = best.predict_proba(X_te)[:, 1]\n',
    '    acc = accuracy_score(y_te, y_pred)\n',
    '    auc = roc_auc_score(y_te, y_prob)\n',
    '    print(f"\\n{chr(61)*55}\\n  {name}\\n{chr(61)*55}")\n',
    '    print(f"Best Params  : {grid.best_params_}")\n',
    '    print(f"CV Accuracy  : {grid.best_score_:.4f}")\n',
    '    print(f"Test Accuracy: {acc:.4f}")\n',
    '    print(f"ROC-AUC Score: {auc:.4f}")\n',
    '    print("\\nConfusion Matrix:")\n',
    '    print(confusion_matrix(y_te, y_pred))\n',
    '    print("\\nClassification Report:")\n',
    '    print(classification_report(y_te, y_pred))\n',
    '    return best, acc, auc'
])

lr_md   = md('f3', ['### Model 1: Logistic Regression\n', '**Tuned:** `C` (regularization strength), `penalty` (l1/l2), `solver` (liblinear)'])
lr_code = code('f4', [
    'lr_model, lr_acc, lr_auc = tune_and_evaluate(\n',
    "    'Logistic Regression',\n",
    '    LogisticRegression(random_state=42, max_iter=1000),\n',
    "    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},\n",
    '    X_train_scaled, y_train, X_test_scaled, y_test\n',
    ')'
])

knn_md   = md('f5', ['### Model 2: K-Nearest Neighbors\n', '**Tuned:** `n_neighbors` (k), `weights` (uniform/distance), `metric` (euclidean/manhattan)'])
knn_code = code('f6', [
    'knn_model, knn_acc, knn_auc = tune_and_evaluate(\n',
    "    'K-Nearest Neighbors',\n",
    '    KNeighborsClassifier(),\n',
    "    {'n_neighbors': [3, 5, 7, 9, 11, 15], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},\n",
    '    X_train_scaled, y_train, X_test_scaled, y_test\n',
    ')'
])

svm_md   = md('f7', [
    '### Model 3: Support Vector Machine (SVM)\n',
    '**Tuned:** `C` (regularization), `kernel` (rbf/linear/poly), `gamma` (kernel coefficient)\n',
    '- `C` controls trade-off between smooth boundary and classifying training points correctly\n',
    '- `rbf` kernel maps data to higher dimensions to find non-linear boundaries\n',
    '- Requires scaled features — already done with StandardScaler'
])
svm_code = code('f8', [
    'svm_model, svm_acc, svm_auc = tune_and_evaluate(\n',
    "    'Support Vector Machine',\n",
    '    SVC(random_state=42, probability=True),\n',
    "    {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['scale', 'auto']},\n",
    '    X_train_scaled, y_train, X_test_scaled, y_test\n',
    ')'
])

compare_md   = md('f9', ['### Model Evaluation & Comparison'])

compare_acc = code('f10a', [
    '# Cell 1 — Accuracy Comparison\n',
    "results = pd.DataFrame({\n",
    "    'Model': ['Logistic Regression', 'KNN', 'SVM'],\n",
    "    'Accuracy': [lr_acc, knn_acc, svm_acc],\n",
    "    'ROC-AUC': [lr_auc, knn_auc, svm_auc]\n",
    "})\n",
    'results = results.sort_values("Accuracy", ascending=False).reset_index(drop=True)\n',
    'print(results.to_string(index=False))\n',
    'plt.figure(figsize=(7, 3))\n',
    "sns.barplot(x='Accuracy', y='Model', data=results, hue='Model', palette='viridis', legend=False)\n",
    'plt.xlim(0.8, 1.0)\n',
    "for i, v in enumerate(results['Accuracy']):\n",
    "    plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)\n",
    "plt.title('Model Accuracy Comparison')\n",
    'plt.tight_layout(); plt.show()'
])

compare_clf = code('f10b', [
    '# Cell 2 — Precision, Recall, F1-Score & Support (Grouped Bar Chart)\n',
    'from sklearn.metrics import classification_report\n',
    'import warnings; warnings.filterwarnings("ignore")\n',
    '\n',
    'models = {\n',
    "    'Logistic Regression': lr_model,\n",
    "    'KNN': knn_model,\n",
    "    'SVM': svm_model\n",
    '}\n',
    '\n',
    'metrics_rows = []\n',
    'for name, m in models.items():\n',
    '    report = classification_report(y_test, m.predict(X_test_scaled), output_dict=True)\n',
    "    for cls in ['0', '1']:\n",
    '        metrics_rows.append({\n',
    "            'Model': name,\n",
    "            'Class': f'Class {cls}',\n",
    "            'Precision': report[cls]['precision'],\n",
    "            'Recall': report[cls]['recall'],\n",
    "            'F1-Score': report[cls]['f1-score'],\n",
    "            'Support': report[cls]['support']\n",
    '        })\n',
    'metrics_df = pd.DataFrame(metrics_rows)\n',
    'print(metrics_df.to_string(index=False))\n',
    '\n',
    '# Grouped bar chart for Precision, Recall, F1\n',
    "plot_df = metrics_df.melt(id_vars=['Model', 'Class'],\n",
    "                           value_vars=['Precision', 'Recall', 'F1-Score'],\n",
    "                           var_name='Metric', value_name='Score')\n",
    "plot_df['Model_Class'] = plot_df['Model'] + '\\n' + plot_df['Class']\n",
    'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
    "for i, cls in enumerate(['Class 0', 'Class 1']):\n",
    "    sub = plot_df[plot_df['Class'] == cls]\n",
    "    sns.barplot(x='Model', y='Score', hue='Metric', data=sub, ax=axes[i], palette='Set2')\n",
    "    axes[i].set_title(f'Precision / Recall / F1 — {cls}')\n",
    "    axes[i].set_ylim(0.7, 1.05)\n",
    "    axes[i].set_xlabel('')\n",
    "    for bar in axes[i].patches:\n",
    "        h = bar.get_height()\n",
    "        if h > 0:\n",
    "            axes[i].text(bar.get_x() + bar.get_width()/2, h + 0.005,\n",
    "                         f'{h:.2f}', ha='center', va='bottom', fontsize=7)\n",
    'plt.tight_layout(); plt.show()'
])

compare_auc = code('f10c', [
    '# Cell 3 — ROC-AUC Comparison\n',
    "auc_df = pd.DataFrame({\n",
    "    'Model': ['Logistic Regression', 'KNN', 'SVM'],\n",
    "    'ROC-AUC': [lr_auc, knn_auc, svm_auc]\n",
    "})\n",
    'auc_df = auc_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)\n',
    'print(auc_df.to_string(index=False))\n',
    'plt.figure(figsize=(7, 3))\n',
    "bars = sns.barplot(x='ROC-AUC', y='Model', data=auc_df, hue='Model', palette='magma', legend=False)\n",
    'plt.xlim(0.8, 1.0)\n',
    "for i, v in enumerate(auc_df['ROC-AUC']):\n",
    "    plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)\n",
    "plt.title('ROC-AUC Score Comparison')\n",
    'plt.tight_layout(); plt.show()'
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
ensemble_md = md('g0', [
    '---\n',
    '## Step 7: Ensemble — Majority Vote\n',
    '\n',
    'Each of the 3 models casts a vote (0 or 1). The class that gets **2 or more votes wins**.\n',
    'No extra library needed — just plain Python.'
])

ensemble_code = code('g1', [
    'import numpy as np\n',
    '\n',
    'lr_preds  = lr_model.predict(X_test_scaled)\n',
    'knn_preds = knn_model.predict(X_test_scaled)\n',
    'svm_preds = svm_model.predict(X_test_scaled)\n',
    '\n',
    '# Majority vote: sum predictions per row; >= 2 means majority says "Purchase"\n',
    'votes = lr_preds + knn_preds + svm_preds\n',
    'ensemble_preds = (votes >= 2).astype(int)\n',
    '\n',
    'ens_acc = accuracy_score(y_test, ensemble_preds)\n',
    'ens_auc = roc_auc_score(y_test, ensemble_preds)\n',
    'print(f"Ensemble Accuracy : {ens_acc:.4f}")\n',
    'print(f"Ensemble ROC-AUC  : {ens_auc:.4f}")\n',
    'print("\\nConfusion Matrix:")\n',
    'print(confusion_matrix(y_test, ensemble_preds))\n',
    'print("\\nClassification Report:")\n',
    'print(classification_report(y_test, ensemble_preds))'
])

ensemble_compare = code('g2', [
    "final_df = pd.DataFrame({\n",
    "    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Ensemble (Majority Vote)'],\n",
    "    'Accuracy': [lr_acc, knn_acc, svm_acc, ens_acc],\n",
    "    'ROC-AUC':  [lr_auc, knn_auc, svm_auc, ens_auc]\n",
    "})\n",
    'final_df = final_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)\n',
    'print(final_df.to_string(index=False))'
])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
save_md   = md('h1', ['---\n', '## Step 8: Save Models & Scaler'])
save_code = code('h2', [
    '# Save all 3 models — app.py will run majority vote at prediction time\n',
    "joblib.dump(lr_model,  'model_lr.pkl')\n",
    "joblib.dump(knn_model, 'model_knn.pkl')\n",
    "joblib.dump(svm_model, 'model_svm.pkl')\n",
    "joblib.dump(scaler,    'scaler.pkl')\n",
    "print('Saved: model_lr.pkl, model_knn.pkl, model_svm.pkl, scaler.pkl')"
])

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE & WRITE
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.8.0'}
    },
    'cells': [
        title, imports,
        # Step 2 — Load
        load_md, load_code, dtypes_code,
        # Step 3 — EDA-1
        eda1_md, eda1_stats, eda1_missing, eda1_gender, eda1_neg,
        eda1_dist_age, eda1_dist_sal, eda1_box_age, eda1_box_sal, eda1_target, eda1_insights,
        # Step 4 — Preprocessing
        prep_md,
        dup_md, dup_code,
        gender_md, gender_code,
        negage_md, negage_code,
        negsal_md, negsal_code,
        impute_md, impute_code,
        encode_md, encode_code,
        corr_md, corr_code,
        drop_md, drop_code,
        outlier_md, outlier_code,
        split_md, split_code,
        # Step 5 — EDA-2
        eda2_md, eda2_stats, eda2_balance, eda2_dist_age, eda2_dist_sal, eda2_box_age, eda2_box_sal,
        eda2_corr, eda2_vs_age, eda2_vs_sal, eda2_insights, eda2_scatter,
        # Step 6 — Models
        model_md, eval_fn,
        lr_md, lr_code,
        knn_md, knn_code,
        svm_md, svm_code,
        compare_md, compare_acc, compare_clf, compare_auc,
        # Step 7 — Ensemble
        ensemble_md, ensemble_code, ensemble_compare,
        # Step 8 — Save
        save_md, save_code,
    ]
}

with open('preprocess_and_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook written: {len(nb["cells"])} cells.')
