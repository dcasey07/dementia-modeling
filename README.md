# Dementia Modeling

## Project Proposal - Dementia Modeling

**Members:** Abel Zemo, Daniel Casey, Jennifer Jones, Teresita Lepasana, and Wilian Uscha

**Decompose the question:**
- Based on the model we develop using this dataset, what is the likelihood of a patient with similar input data running the risk of having dementia?

**Identify Data Sources:**
- Using a kaggle dataset that scraped data from PUBMED, Online research sources, NHS, Google scholar and consultation with healthcare professionals.
- Data includes 1000 entries
- https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data

**Pipeline to Retrieve and Clean:**
- Data In:
  - CSV file
  - All integer based data
- Data Cleaning/Processing:
  - Filter outliers using Boxplots, IQRs of each column of data
  - Combination of categorical and integer based data
    - `pd.get_dummies()` will be necessary for converting categorical data

**Trend Analysis:**
- What features yield the most common trends for patients that have dementia?
- Correlation analysis for each feature

**Limitations of the exercise / Tell a story:**
- 1000 entries, low entry dataset - 80/20 train test split
- What conditions increase the likelihood of having dementia?
- What correlations are present between the features and the target?

**Machine Learning:**
- Supervised Learning:
  - Target: 0 - No dementia; 1 - Yes dementia
- Potential models:
  - Random Forest (Visualize Feature Importance)
  - Logistic Regression - Wilian
  - Neural Network - Teresita
  - KNN
  - XGBoost - Daniel

**Rubric Considerations**

- Data and data delivery - Kaggle input csv > Pandas/Jupyter notebook > Cleaned output csv
- Back End (ETL) - Potentially SQLite (smaller dataset)
- Visualizations - Matplotlib
- Group presentations - Google Slides
  - Overview of dataset source and objective
  - Exploratory Data Analysis
  - Outlier Detections/Model Optimizations
  - Modeling Implementation/Results/Group’s Model of Choice
  - Flask/APIs (Run the model through python)
  - Create website that can support input data to test the model

