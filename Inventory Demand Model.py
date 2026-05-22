#Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# --------------------------------Prepare Data --------------------------------

#1. Import synthetic data
Retail_store_data = "/Users/alanngungi/Desktop/Important/Work/Data Eng Portfolio/Inventory Demand Predictor/sales_data.csv"
df = pd.read_csv(Retail_store_data)
print(df.shape)
df.head()

#2. Explore Data
#Investigate data for null values
df.info()

#Create year, month, day column to replace Date column in dataset
parts = df["Date"].str.split("-", n = 3, expand = True)
df["year"]= parts[0].astype('int')
df["month"]= parts[1].astype('int')
df["day"]= parts[2].astype('int')

df.drop('Date', axis=1, inplace=True)
df.head()

#Investigate distribution of target vector
plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.distplot(df['Demand'])

plt.subplot(1, 2, 2)
sns.boxplot(df['Demand'])
plt.show();

#Remove outlier data according to boxplot
df = df[df["Demand"]<230]

#Drop unneccesary features and identify correlations
cols = ["Store ID", "Product ID", "Weather Condition"]
df.drop(columns = cols, inplace = True)

corr = df.select_dtypes("number").corr()
corr

#Bar graph visualisation of mean demand according to specific features
columns = [ 
    "Inventory Level",
    "Units Sold", 
    "Units Ordered", 
    "Price", 
    "Discount", 
    "Promotion", 
    "Competitor Pricing",
    "Epidemic",
    "Demand",
    "year",
    "month",
    "day"
]
plt.subplots(figsize=(15, 5))
plt.subplot(1, 3, 1)
df[columns].groupby(df["Category"]).mean()['Demand'].plot.bar()
plt.ylabel("Demand")

plt.subplot(1, 3, 2)
df[columns].groupby(df["Region"]).mean()['Demand'].plot.bar()
plt.ylabel("Demand")

plt.subplot(1, 3, 3)
df[columns].groupby(df["Seasonality"]).mean()['Demand'].plot.bar()
plt.ylabel("Demand");

#Line graph visualisation of mean demand by Day of the month
df[columns].groupby("day").mean()['Demand'].plot()
plt.xlabel("Day of the Month")
plt.ylabel("Demand");

#3. Split data
#Split data, horizontally
features = df.drop(["Demand"], axis=1)
target = df["Demand"].values

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                  test_size = 0.2,
                                                  random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# -------------------------------- Build Model --------------------------------
#1. Calculate baseline to beat
y_mean = y_train.mean()
y_pred_baseline = [y_mean]*len(y_train)

print('Baseline MAE : ', mae(y_train, y_pred_baseline))

#2. Train regression model pipelines on split data
# 2.1. LinearRegression():
# Build Linear regression model pipeline
model_lr = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LinearRegression()
)
model_lr.fit(X_train, y_train)
print("✅ OHE-LinearRegression Model pipeline training complete!")

#Calculate training and test error for LinearRegression model evaluation
train_preds_lr = model_lr.predict(X_train)
print('Training Error : ', mae(y_train, train_preds_lr))



#2.2. Lasso():
#Build Lasso model pipeline
model_l = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Lasso()
)
model_l.fit(X_train, y_train)
print("✅ OHE-Lasso Model pipeline training complete!")

#Calculate test error for Lasso model evaluation
train_preds_l = model_l.predict(X_train)
print('Training Error : ', mae(y_train, train_preds_l))



#2.3. Ridge():
#Build Ridge model pipeline
model_r = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    Ridge(fit_intercept=True)
)
model_r.fit(X_train, y_train)
print("✅ OHE-Ridge Model pipeline training complete!")

#Calculate test error for Ridge model evaluation
train_preds_r = model_r.predict(X_train)
print('Training Error : ', mae(y_train, train_preds_r))



# -------------------------------- Model Evaluation/Results Communication --------------------------------

#1. LinearRegression():
#Calculate test error for LinearRegression model evaluation
test_preds_lr = model_lr.predict(X_test)
print('Test Error : ', mae(y_test, test_preds_lr))

#Obtain LinearRegression model info
intercept_lr = model_lr.named_steps['linearregression'].intercept_
coef_lr = model_lr.named_steps['linearregression'].coef_
print(intercept_lr)
print(coef_lr)

#2. Lasso():
#Calculate test error for Lasso model evaluation
test_preds_l = model_l.predict(X_test)
print('Test Error : ', mae(y_test, test_preds_l))

#Obtain Lasso model info
intercept_l = model_l.named_steps['lasso'].intercept_
coef_l = model_l.named_steps['lasso'].coef_
print(intercept_l)
print(coef_l)

#3. Ridge():
#Calculate test error for Ridge model evaluation
train_preds_r = model_r.predict(X_train)
print('Training Error : ', mae(y_train, train_preds_r))

test_preds_r = model_r.predict(X_test)
print('Test Error : ', mae(y_test, test_preds_r))

#Obtain Ridge model info
intercept_r = model_r.named_steps['ridge'].intercept_
coef_r = model_r.named_steps['ridge'].coef_
print(intercept_r)
print(coef_r)