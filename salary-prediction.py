import pandas as pd
import numpy as np


df = pd.read_csv('salary-job-data.csv')



df.head()



# lower case the columns and replace whitespace with underscores.
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head()



df.dtypes



df = df.dropna()

# check the shape to confirm how many rows remain
print(df.shape)

# (optional) confirm there are no missing values
print(df.isnull().sum())



# isolate the columns containing strings
strings = list(df.dtypes[df.dtypes == 'object'].index)
strings



# now all strings should be lower and camel cased
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()


# # EDA


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()



import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')



# plot the count of salaries in the data set
sns.histplot(df.salary, bins=36)



df.head()



# mean salary for females
salary_female = df[df.gender == 'female'].salary.mean()
salary_female = round(salary_female, 2)



# mean salary for males
salary_male = df[df.gender == 'male'].salary.mean()
salary_male = round(salary_male, 2)



f'The mean salary of females: ${salary_female}', f'the mean salary of males: ${salary_male}'



print(df.groupby('gender')['salary'].describe())



plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='salary', data=df)
plt.title('Salary Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Salary (USD)')
plt.show()


# # Feature Importance


from sklearn.metrics import mutual_info_score



def mutual_info_churn_score(series):
    return mutual_info_score(series, df.salary)



categorical = ['gender', 'education_level', 'job_title']



# job title is more important than education and gender
mi = df[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)



top_titles = df['job_title'].value_counts().head(10).index
df_top = df[df['job_title'].isin(top_titles)]



plt.figure(figsize=(12, 6))
sns.swarmplot(x='job_title', y='salary', data=df_top, size=4, alpha=0.7)
plt.title('Salary Clusters by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Salary (USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # Regression


# where do we have null values and how many are there?
df.isnull().sum()



from sklearn.model_selection import train_test_split



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)



len(df_full_train), len(df_test)



df_train, df_val =  train_test_split(df_full_train, test_size=0.25, random_state=1)



len(df_train), len(df_test), len(df_val)



df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



y_train = df_train.salary.values
y_val = df_val.salary.values
y_test = df_test.salary.values



del df_train['salary']
del df_val['salary']
del df_test['salary']



from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer



# Convert dataframe to list of dictionaries
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')



dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)



X_train.shape, X_val.shape



model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_val)



from sklearn.metrics import mean_squared_error



# calculate the root mean square error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print('Validation RMSE:', rmse)



# standard deviation of salaries
np.std(y_val)



plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred, alpha=0.7)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red')
plt.show()



# Not great, let's see if a decision tree it better



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# try a simple tree first (no tuning)
dt = DecisionTreeRegressor(random_state=1)
dt.fit(X_train, y_train)

# predict on validation
y_pred_dt = dt.predict(X_val)

# evaluate RMSE
rmse_dt = np.sqrt(mean_squared_error(y_val, y_pred_dt))
print('Decision Tree RMSE:', rmse_dt)



y_pred_lr = model.predict(X_val)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))

print('Linear Regression RMSE:', rmse_lr)
print('Decision Tree RMSE:', rmse_dt)



for d in [1, 2, 3, 4, 5, 6, 10, 15, 20]:
    dt = DecisionTreeRegressor(max_depth=d, random_state=1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'max_depth={d:>4}  RMSE={rmse:.2f}')



# max_depth = 6 where RMSE=16131.49 is the best let's plot it



depths = [1, 2, 3, 4, 5, 6, 10, 15, 20]
rmses = []

for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmses.append(rmse)

plt.plot(depths, rmses, marker='o')
plt.title('Decision Tree Depth vs RMSE')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.show()



# let's tune min_sample_leaf
for leaf_size in [1, 2, 5, 10, 20, 50]:
    dt = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=leaf_size,
        random_state=1
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'min_samples_leaf={leaf_size:>3}  RMSE={rmse:.2f}')



# let's plot these to visualize
leaf_sizes = [1, 2, 5, 10, 20, 50]
rmses = []

for leaf_size in leaf_sizes:
    dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=leaf_size, random_state=1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmses.append(rmse)

plt.plot(leaf_sizes, rmses, marker='o')
plt.title('min_samples_leaf vs RMSE')
plt.xlabel('min_samples_leaf')
plt.ylabel('RMSE')
plt.show()



dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=1)
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_pred_val = dt.predict(X_val)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"Train RMSE: {rmse_train:.2f}")
print(f"Validation RMSE: {rmse_val:.2f}")



features = dv.get_feature_names_out()
importances = dt.feature_importances_

df_importance = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(df_importance.head(10))



plt.figure(figsize=(8, 5))
plt.barh(df_importance['feature'][:10], df_importance['importance'][:10])
plt.gca().invert_yaxis()
plt.title('Top 10 Feature Importances (Decision Tree)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()



X_test = dv.transform(df_test.to_dict(orient='records'))
y_pred_test = dt.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print('Test RMSE:', rmse_test)



from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=dv.get_feature_names_out(), filled=True, max_depth=3)
plt.show()



i = 10  # any index you want

# grab the row from the validation set
sample = df_val.iloc[i]

sample



actual_salary = y_val[i]
print(f'Actual salary: ${actual_salary:,.2f}')



# turn it into a single-record list of dicts
sample_dict = sample.to_dict()

# transform using the fitted DictVectorizer
X_sample = dv.transform([sample_dict])



predicted_salary = dt.predict(X_sample)[0]
print(f'Predicted salary: ${predicted_salary:,.2f}')



import numpy as np

for i in np.random.choice(len(df_val), 5, replace=False):
    record = df_val.iloc[i].to_dict()
    actual = y_val[i]
    pred = dt.predict(dv.transform([record]))[0]
    print(f"Row {i} | Actual: ${actual:,.0f} | Predicted: ${pred:,.0f} | Difference: ${pred-actual:,.0f}")
