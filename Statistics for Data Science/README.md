# Malignant Melanoma Survival Analysis

## Overview
This project analyses survival outcomes in malignant melanoma patients using statistical methods in R.  
The aim was to explore which factors influence survival time and identify patterns in patient data.

The dataset includes patient information such as survival time, age, tumor thickness, ulceration status, and gender.

## Key Findings
- Survival time shows high variability, ranging from a few days to over 15 years  
- Tumor thickness and age have a weak negative relationship with survival time  
- Women tend to have slightly better survival outcomes than men  
- Tumor thickness differs significantly between males and females  
- Age does not show a significant difference between genders  

## Methods Used
- Summary statistics  
- Data visualisation (histograms, boxplots, scatterplots)  
- Correlation analysis  
- Linear regression models  
- Hypothesis testing:
  - Chi-square test  
  - t-test  
  - Mann–Whitney U test  

## What I Did
- Cleaned and prepared the dataset  
- Explored distributions of key variables  
- Investigated relationships between survival time, age, and tumor thickness  
- Applied statistical tests to analyse gender-based differences  

## Limitations
- Weak relationships between variables limit predictive power  
- Historical dataset (1962–1977), so results may not reflect current trends  
- Missing external factors (e.g. treatment type, lifestyle)  

## How to Run
Clone the repository and run the analysis in RStudio.

```bash
git clone https://github.com/solaakande/Data_Projects.git
