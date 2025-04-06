#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 06:33:53 2025

@author: yajie
"""



import os
import pandas as pd

# Change to the desired directory
os.chdir("HackUSF-2025/data_ana")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice default style
sns.set(style="whitegrid")

# Load your data (adjust the path accordingly)
clinical = pd.read_csv("HackUSF-2025/healthcare-data/clinical.csv")

# Select relevant columns
df = clinical[[
    'cases_submitter_id', 
    'cases_disease_type',
    'demographic_age_at_index',
    'demographic_country_of_residence_at_enrollment',
    'demographic_ethnicity',
    'demographic_gender',
    'demographic_race',
    'demographic_vital_status',
    'diagnoses_days_to_diagnosis',
    'treatments_treatment_type'
]].drop_duplicates(subset='cases_submitter_id')

# Remove extreme values or missing for clarity
df = df[df['demographic_age_at_index'].notna()]
df = df[df['demographic_age_at_index'] < 120]  # Remove unrealistic ages


# =============================================================================
# # Plots
# =============================================================================


plt.figure(figsize=(10, 5))
sns.histplot(df['demographic_age_at_index'], bins=30, kde=True)
plt.title("Distribution of Patient Age")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("age_his.png", dpi=300, bbox_inches='tight')
plt.show()



# Normalize gender column
df['gender_cleaned'] = df['demographic_gender'].str.strip().str.lower()
# Count
gender_counts = df['gender_cleaned'].value_counts()

# Define color map for lowercase values
color_map = {
    'female': '#ffb6c1',  # pastel pink
    'male': '#add8e6'     # pastel blue
}
colors = [color_map.get(g, '#d3d3d3') for g in gender_counts.index]  # gray if unmatched
# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index.str.title(), autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Overall Gender Distribution")
plt.axis('equal')
plt.savefig("gender_pie.png", dpi=300, bbox_inches='tight')
plt.show()


# Normalize and clean race values
df['race_cleaned'] = df['demographic_race'].str.strip().str.lower()
# Replace 'unknown' and 'not reported' with NA
df['race_cleaned'] = df['race_cleaned'].replace({'unknown': pd.NA, 'not reported': pd.NA})
# Get race counts, including NaNs
race_counts = df['race_cleaned'].value_counts(dropna=False)
# Replace NaN with readable label
labels = race_counts.index.to_series().fillna("Unknown / Not Reported").str.title()
# Generate pastel colors
colors = sns.color_palette("pastel")[0:len(race_counts)]
# Plot the pie chart
plt.figure(figsize=(7, 7))
plt.pie(
    race_counts,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)
plt.title("Race Distribution")
plt.axis('equal')
plt.tight_layout()
plt.savefig("race_pie.png", dpi=300, bbox_inches='tight')
plt.show()