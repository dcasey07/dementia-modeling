# Dementia Modeling

## Project Proposal - Dementia Modeling

**Members:** Abel Zemo, Daniel Casey, Teresita Lepasana, and Wilian Uscha

**Purpose of the project:**
- Based on the model we develop using this dataset, what is the likelihood of a patient with similar input data running the risk of having dementia?

**Data Sources:**
- Using a kaggle dataset that scraped data from PUBMED, Online research sources, NHS, Google scholar and consultation with healthcare professionals.
- Data includes 1000 entries
- https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset/data

**Pipeline to Retrieve and Clean:**
- Data In:
  - CSV file
  - Combination of integer based data and categorical data
- Data Cleaning/Processing:
  - Filter outliers using Boxplots, IQRs of each column of data
  - Combination of categorical and integer based data
    - Remapping of binary classifcation features  
    - `pd.get_dummies()` will be necessary for converting categorical data
    - Removed post-diagnosis features and redundancies (Diabetic, Prescription, Dosage in mg)
   
**- Features Used in Modeling:**
  - AlcoholLevel
  - HeartRate
  - BloodOxygenLevel
  - BodyTemperature
  - Weight (in kg)
  - MRI_Delay
  - Age
  - Education_Level
  - Dominant_Hand
  - Gender
  - Family_History
  - Smoking_Status
  - APOE_ε4 (Alzheimer's Genetic Risk Factor)
  - Physical_Activity
  - Depression_Status
  - Cognitive_Test_Scores
  - Medication_History
  - Nutrition_Diet
  - Sleep_Quality
  - Chronic_Health_Conditions
 
**- Target:**
  - Dementia (1: Positive, 0: Negative) 

**Trend Analysis:**
- What features yield the most common trends for patients that have dementia?
- Correlation analysis for each feature

**Limitations of the exercise / Tell a story:**
- 1000 entries, low entry dataset - 80/20 train test split
- What conditions increase the likelihood of having dementia?
- What correlations are present between the features and the target?

**Models in the Project:**
- Random Forest
- Logistic Regression
- Neural Network
- XGBoost

**Database**
Information used in the modeling was output into a sqlite database after preprocessing, but before encoding.

**Deployment:**
The XGBoost model was deployed on Sagemaker and later used to create a webform application that attempts to predict if the input data reflects a positive or negative case of Dementia
