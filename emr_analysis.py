# %%
##  Exploratory Analysis on EMR Data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Optional: prettier plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the Excel file
file_path = r"C:\Users\emudr\Desktop\data\synthetic_emr_data.csv"
df = pd.read_csv(file_path)

# Peek at the data
df.head()

# %%
# Shape of dataset
print("Shape:", df.shape)

# Data types and missing values
df.info()

# Descriptive statistics for numeric fields
df.describe()

# Check for missing values
df.isnull().sum()


# %% [markdown]
# 

# %%
# Count unique values in each column
df.nunique()

# Check class distributions (example: Gender, Diagnosis, etc.)
if 'gender' in df.columns:
    df['gender'].value_counts().plot(kind='bar', title='Gender Distribution')
    plt.show()

if 'medical_condition' in df.columns:
    df['medical_condition'].value_counts().plot(kind='barh', title='Top Diagnoses')
    plt.show()

if 'ethnicity' in df.columns:
    sns.histplot(df['ethnicity'], kde=True, bins=20)
    plt.title("Ethnicity")
    plt.show()

if 'age' in df.columns:
    sns.histplot(df['age'], kde=True, bins=20)
    plt.title("Age Distribution")
    plt.show()



# %%
# Overall counts
condition_counts = df['medical_condition'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=condition_counts.values, y=condition_counts.index, palette='viridis')
plt.title("Prevalence of Medical Conditions")
plt.xlabel("Number of Patients")
plt.ylabel("Medical Condition")
plt.tight_layout()
plt.show()


# %%
# Countplot grouped by gender
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='medical_condition', hue='gender', order=condition_counts.index, palette='Set2')
plt.title("Prevalence of Medical Conditions by Gender")
plt.xlabel("Count")
plt.ylabel("Medical Condition")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='medical_condition', y='age', palette='coolwarm')
plt.title("Age Range per Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '21-40', '41-60', '61-80', '81+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Count top conditions per age group
age_condition = df.groupby(['age_group', 'medical_condition']).size().reset_index(name='count')

plt.figure(figsize=(12, 6))
sns.barplot(data=age_condition, x='age_group', y='count', hue='medical_condition')
plt.title("Top Medical Conditions by Age Group")
plt.ylabel("Count")
plt.xlabel("Age Group")
plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# 

# %%
# Grouping and counting
ethnicity_conditions = df.groupby(['ethnicity', 'medical_condition']).size().reset_index(name='count')

# Display the table
ethnicity_conditions.head()

# Pivot the data for heatmap
pivot_table = ethnicity_conditions.pivot(index='ethnicity', columns='medical_condition', values='count').fillna(0)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='g', cmap='Blues')
plt.title('Medical Conditions by Ethnicity')
plt.ylabel('Ethnicity')
plt.xlabel('Medical Condition')
plt.tight_layout()
plt.show()

# %%
# Total counts per ethnicity (for normalization)
ethnicity_totals = df['ethnicity'].value_counts().to_dict()

# Count of each condition per ethnicity
ethnicity_conditions = df.groupby(['ethnicity', 'medical_condition']).size().reset_index(name='count')

# Add total per ethnicity to compute % relative to ethnicity group size
ethnicity_conditions['percent'] = ethnicity_conditions.apply(
    lambda row: (row['count'] / ethnicity_totals[row['ethnicity']]) * 100,
    axis=1
)

# Pivot to make heatmap-friendly table
pivot_percent = ethnicity_conditions.pivot(index='ethnicity', columns='medical_condition', values='percent').fillna(0)

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_percent, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': '% of People in Ethnicity Group'})
plt.title('Percent of Medical Conditions by Ethnicity')
plt.ylabel('Ethnicity')
plt.xlabel('Medical Condition')
plt.tight_layout()
plt.show()


# %%
# Create age bins
bins = [0, 18, 30, 45, 60, 75, 100]
labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Group by age group and condition
risk_profiles = df.groupby(['age_group', 'medical_condition']).size().reset_index(name='count')

# Pivot for heatmap
risk_pivot = risk_profiles.pivot(index='age_group', columns='medical_condition', values='count').fillna(0)

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(risk_pivot, annot=True, fmt='g', cmap='Oranges')
plt.title('Medical Condition Risk by Age Group')
plt.xlabel('Medical Condition')
plt.ylabel('Age Group')
plt.tight_layout()
plt.show()


# %%
# Create age bins
bins = [0, 18, 30, 45, 60, 75, 100]
labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# Count per age group and medical condition
risk_profiles = df.groupby(['age_group', 'medical_condition']).size().reset_index(name='count')

# Total per age group (for normalization)
age_totals = df['age_group'].value_counts().to_dict()

# Calculate percentage
risk_profiles['percent'] = risk_profiles.apply(
    lambda row: (row['count'] / age_totals[row['age_group']]) * 100,
    axis=1
)

# Pivot for heatmap
risk_pivot = risk_profiles.pivot(index='age_group', columns='medical_condition', values='percent').fillna(0)

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(risk_pivot, annot=True, fmt='.1f', cmap='Oranges', cbar_kws={'label': '% of People in Age Group'})
plt.title('Percent of Medical Conditions by Age Group')
plt.xlabel('Medical Condition')
plt.ylabel('Age Group')
plt.tight_layout()
plt.show()



