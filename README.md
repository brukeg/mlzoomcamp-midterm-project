# Machine Learning Zoomcamp Midterm Project 2025 
This project attempts to answer, how can you predict salary given some set of limited employee profile data.

## Predicting Salary
Using two models: **Linear Rergressor** and **Decision Tree Regressor** we will then package-up the best performing model as a simple Flask API that can be run inside Docker.  

## Dataset
I got the data set from Kaggle, and what's notable about this data is that it is synthetic--for education only--it is absolutely not real data at all. Each row represents one (fake as all heck) employee.

You can read more about the data on Kaggle: https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer?resource=download

**Columns**

| Column | Type | Description |
|:--------|:------|:------------|
| `age` | float | Employee’s age |
| `gender` | string | `"male"` or `"female"` |
| `education_level` | string | `"bachelors"`, `"masters"`, or `"phd"` |
| `job_title` | string | Employee’s job title |
| `years_of_experience` | float | Years of work experience |
| `salary` | float | Annual salary (target variable) |

## Approach

1. **Data Cleaning**  
   - Lowercase all column names and string values  
   - Replace spaces with underscores  
   - Drop missing rows  

2. **EDA & Feature Importance**  
   - Explore relationships between salary and gender, education, and job title  
   - Compute mutual information for categorical features  
   - Plot of actual salary vs predicted salary

3. **Modeling**  
   - Compare **Linear Regression** (baseline) vs. **Decision Tree Regressor**  
   - Tune and select the best hyperparameters for the Decision Tree Regressor (`max_depth` and `min_samples_leaf`)

4. **Evaluation**  
   - RMSE on train/val/test splits (random_state=1)

5. **Deployment**  
   - Save `(DictVectorizer, model)` as `model_decision_tree.bin`  
   - Serve predictions from a small **Flask API** (`predict.py`)  

**Current results**

With `max_depth=6` and `min_samples_leaf=2`)
- Train RMSE: ~10,734  
- Val RMSE: ~15,188  
- Test RMSE: ~14,595

--
# Getting Started


1. Clone the repo
```bash
git clone https://github.com/<your-username>/mlzoomcamp-midterm-project.git
cd mlzoomcamp-midterm-project
```

2. Install dependencies
```bash
pip install --user pipenv
```

3. Train the model
<!-- If you don't change the model you should only need to do this once. -->
```bash
pipenv run python train.py
```

4. Build the Docker image.
```bash
docker build -t salary-api .
```

5. Run the web-service
```bash
docker run -rm -p 9696:9696 salary-api
```

6. Make a test request
```bash
curl -s -X POST http://127.0.0.1:9696/predict \
-H "Content-Type: application/json" \
-d '{"age":44,"gender":"female","education_level":"masters","job_title":"senior_data_engineeR","years_of_experience":15}' \
| jq
```

You should see something like the following:
```bash
{
  "salary_prediction": 123333.33333333333
}
```
-- 
# Make your own prediction

### Data Validation
- `age`: must be between 23 and 53
- `years_of_experience`: must be from 0-25 and ≤ `age` - 18
- `education_level`" must be one of: `bachelors`, `masters`, or `phd`
- `gender`: can be `male`, or `female`
- `job_title`: must have been seen in the training data (see examples below)

**sample job_title list to chose from (full list in the notebook):**
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

mlzoomcamp-midterm-project/
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