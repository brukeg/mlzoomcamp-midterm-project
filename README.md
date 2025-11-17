# mlzoomcamp-midterm-project
ML Zoomcamp 2025 Midterm Project

## Salary Prediction (ML Zoomcamp – Midterm)
Predict an employee’s salary from basic profile features using a Decision Tree Regressor, packaged with a small Flask API for local or containerized serving.

## Dataset
Synthetic dataset for education only. One row per employee.

You can read more about the data on Kaggle: https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer?resource=download

**Columns**

- `age` (float)
- `gender` (object)
- `education_level` (object)
- `job_title` (object)
- `years_of_experience` (float)
- `salary` (float) – target

Typical ranges in this dataset:
- `age`: 23 – 53
- `years_of_experience`: 0 – 25

## Approach
- Reads a CSV with columns:
  - Numeric: age (float), years_of_experience (float), salary (float)
  - Categorical: gender (str), education_level (str), job_title (str),
- Clean the data (lowercases column names and string values, drops NaNs)
- Exploratory Analysis
- Feature Importance
- Split data into train/validation/test with a fixed random_state
- Encode features with DictVectorizer (fit on train only)
- **Model trained:** LinearRegressor
- Plot of actual vs predicted
- **Model trained:** DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=1)
- Compare the two models and pick one to deploy as a simple web-service
- Reports RMSE on train/val/test splits of the deployed model
- Saves a single pickle file containing (dv, model): model_decision_tree.bin
- Flask service that loads (dv, model) from model_decision_tree.bin
- Sserve a POST /predict endpoint returning a salary prediction.

**Current results**

- Train RMSE: ~10,734  
- Val RMSE: ~15,188  
- Test RMSE: ~14,595

### Data Validation (for curl requests)
- age must be 23 and 53 (based on training data)
- years_of_experience must be 0 and 25 and ≤ age - 18
- education_level must be one of: `bachelors`, `masters`, or `phd`
- gender: `male`, or `female`
- job_title: must have been seen in training (see examples below)

**sample job_title list to chose from (full list in notebook):**
`account_manager`, `administrative_assistant`, `business_analyst`, `business_intelligence_analyst`, `ceo`, `chief_data_officer`, `copywriter`, `customer_service_manager`, `customer_service_rep`, `customer_success_manager`, `customer_success_rep`, `data_analyst`, `director_of_business_development`, `director_of_finance`, `director_of_hr`, `director_of_human_capital`, `director_of_human_resources`, `director_of_marketing`, `director_of_operations`, `director_of_sales`, `director_of_sales_and_marketing`, `event_coordinator`, `financial_advisor`, `financial_analyst`, `financial_manager`, `graphic_designer`, `help_desk_analyst`, `hr_generalist`, `it_manager`, `it_support`

Input JSON 
```json
{
  "age": 31.0,
  "gender": "male",
  "education_level": "bachelor",
  "job_title": "data_engineer",
  "years_of_experience": 5.0
}
```

Response JSON:
```json
{
  "salary_prediction": 123456.78
}
```

## Repo layout

mlzoomcamp-midterm-project
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
├── model_decision_tree.bin
├── predict.py
├── salary-job-data.csv
├── salary-prediction.ipynb
├── salary-prediction.py
└── train.py

## Local setup (Pipenv)

### Requirements
what are the requirements?
pipenv installed?

```bash
pipenv install
pipenv run python train.py    # prints RMSEs and writes model_decision_tree.bin
docker build -t salary-api .
docker run --rm -p 9696:9696 salary-api
```

#### Example Request
```bash
curl -s -X POST http://127.0.0.1:9696/predict \
-H "Content-Type: application/json" \
-d '{"age":44,"gender":"female","education_level":"Bachelors","job_title":"senior_data_engineeR","years_of_experience":20}' \
| jq
```

#### Output
```bash
{
  "salary_prediction": 174375.0
}
```