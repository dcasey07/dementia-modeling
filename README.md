# Diabetes Modeling

## Project Proposal - Diabetes Modeling

**Members:** Abel Zemo, Daniel Casey, Jennifer Jones, Teresita Lepasana, and Wilian Uscha

**Decompose the question:**
- Based on the model we develop using this dataset, what is the likelihood of a patient with similar input data running the risk of having diabetes?

**Identify Data Sources:**
- Using a dataset pulled from the National Institute of Diabetes and Digestive and Kidney Diseases, this dataset includes records obtained for the purpose of diagnostically predicting  whether a patient has diabetes, based on certain diagnostic measurements included in the dataset.
- Data includes 768 entries
- All patients in the dataset are females, at least 21 years of age, and of Pima Indian heritage
  - https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

**Pipeline to Retrieve and Clean:**
- Data In:
  - CSV file
  - All integer based data
- Data Cleaning/Processing:
  - Filter outliers
  - Boxplots, IQRs of each column of data
  - Cleaning out 0’s where a result of 0 is impossible (blood pressure, skin hardness, etc.)

**Trend Analysis:**
- What columns yield the most common trends for women 21 years or older that have diabetes?

**Limitations of the exercise / Tell a story:**
- Data is limited to females, 21 years or older, with similar heritage
- Most columns with 0’s have to be treated as null values

**Machine Learning:**
- Supervised Learning:
  - Target: 0 - No diabetes; 1 - Yes diabetes
- Potential models:
  - Oversampling (Normalize the classes to balance the 0’s and 1’s)
  - Random Forest (Visualize Feature Importance)
  - Logistic Regression
  - Neural Network
  - KNN

**Rubric Considerations**

- Data and data delivery - Kaggle input csv > Pandas/Jupyter notebook > Cleaned output csv
- Back End (ETL) - Potentially SQLite or SQL (smaller dataset)
  - Not sure how necessary this is given the size and scope of the data
- Visualizations - Matplotlib
- Group presentations - Google Slides
  - Overview of dataset source and objective
  - Exploratory Data Analysis
  - Outlier Detections/Model Optimizations
  - Modeling Implementation/Results/Group’s Model of Choice
