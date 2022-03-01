# modules to import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# helpful functions
def space():

    print("\n\n")


# 1. 
"""
1. Load the data file.
"""
# reading file
path_from_Project = "./Files/hour.csv"
path_from_Bosch = "./Project/Bike-Sharing Demand Analysis/Files/hour.csv"
df = pd.read_csv(path_from_Bosch)     #CHANGE DEPENDING ON WHERE U ARE OPENING SOLUTION FROM
df = pd.DataFrame(df)

# file description and details 
print("File Details\n")
print("===============================================================")
print(df.head(5))
space()
print(df.info())
space()
print(df.describe())
print("===============================================================")
space()
for column in df.columns:
    print(df[column].value_counts())
    space()
print("===============================================================")


# 2.
"""
2. Check for null values in the data and drop records with NAs.
"""
# checking for NA values in each column
print(df.isna().sum(axis = 0)) 
space()

# 3.
"""
3. Sanity checks:
    1. Check if registered + casual = cnt for all the records. If not, the row is junk and should be dropped.
	2. Month values should be 1-12 only
	3. Hour values should be 0-23
"""
# SANTITY CHECK. Checking if registered + casual = cnt for all the records. 
# If not, the row is junk and should be dropped.
df["valid-count"] = np.sum(df["casual"] + df["registered"]) != df["cnt"]
print(df["valid-count"].value_counts()) 
space()
# as all are True no row will be deleted

# removing the rows that dont follow 'registered + casual = cnt'
df = df[df["valid-count"] != False]

# remove unwanted column - "valid-count"
df.drop("valid-count", axis = 1, inplace = True)

# SANTITY CHECK. Checking if Month values are 1-12 only.
print("months: ", np.unique(df["mnth"]))
space()

# SANTITY CHECK. Checking if Hour values are 0-23 only.
print("hours: ", np.unique(df["hr"]))
space()

# changing day of the week numbers from 0-6 to 1-7
print(pd.unique(df["weekday"]))
for i in range(0, 7):
	df["weekday"].replace(6-i, 7-i, inplace = True)
print(pd.unique(df["weekday"]))

# 4.
"""
4. The variables ‘casual’ and ‘registered’ are redundant and need to be dropped. ‘Instant’ is the index and needs
   to be dropped too. The date column dteday will not be used in the model building, and therefore needs to be 
   dropped. Create a new dataframe named inp1.
"""
#drop columns- "casual", "registered", "Instant", "dteday"
inp1 = df.drop(["casual", "registered", "instant", "dteday"], axis = 1).copy()
inp1 = pd.DataFrame(inp1)
print(inp1.head(10))
space()
print(inp1.info())
space()


# 5.
"""
5. Univariate analysis: 
	> Describe the numerical fields in the dataset using pandas describe method.
	> Make density plot for temp. This would give a sense of the centrality and the spread of the distribution.
	> Boxplot for atemp 
		> Are there any outliers?
	> Histogram for hum
		> Do you detect any abnormally high values?
	> Density plot for windspeed
	> Box and density plot for cnt – this is the variable of interest 
		> Do you see any outliers in the boxplot? 
		> Does the density plot provide a similar insight?
"""

print("---------------------------UNIVARIATE ANALYSIS---------------------------")
# Univariate analysis: Describe the numerical fields in the dataset using pandas describe method.
print(inp1.describe())
space()

# Univariate analysis: Make density plot for temp. This would give a sense of the centrality and the spread of the 
# distribution.
style.use("ggplot")
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
sns.distplot(inp1["temp"], bins = 41, hist_kws = {"edgecolor" : "black"})
plt.title("Temperature Distribution")
# OR
plt.subplot(1, 2, 2)
inp1["temp"].plot.density()

plt.show()

# Univariate analysis: Boxplot for atemp.
style.use("ggplot")
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
sns.boxplot("atemp", data = inp1)
plt.title("Temperature felt boxplot")
# OR
plt.subplot(1, 2, 2)
inp1["atemp"].plot.box()

plt.show()
# Univariate analysis: Are there any outliers?
# ANSWER: NO

# Univariate analysis: Histogram for hum
style.use("ggplot")
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data = inp1, x = "hum")
plt.title("Humidity Distribution")
# OR
plt.subplot(1, 2, 2)
inp1["atemp"].plot.hist()
#try annotating frequency later

plt.show()
# Univariate analysis: Do you detect any abnormally high values?
# ANSWER: NO

# Univariate analysis: Density plot for windspeed.
style.use("ggplot")
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
sns.distplot(inp1["windspeed"])
plt.title("Wind speed distribution plot")
# OR
plt.subplot(1, 2, 2)
inp1["windspeed"].plot.density()

plt.show()

# Univariate analysis: Box and density plot for cnt – this is the variable of interest 
style.use("ggplot")
plt.figure(figsize = (14, 6))

plt.subplot(2, 2, 1)
sns.boxplot("cnt", data = inp1)

plt.subplot(2, 2, 2)
inp1["cnt"].plot.box()

plt.subplot(2, 2, 3)
sns.distplot(inp1["cnt"])

plt.subplot(2, 2, 4)
inp1["cnt"].plot.density()

plt.show()
# Univariate analysis: Do you see any outliers in the boxplot? 
# ANSWER: YES> A LOT
# Univariate analysis: Does the density plot provide a similar insight?? 
# ANSWER: YES (sort of?) #CHECK#


# 6.
"""
6. Outlier treatment:  
	1. Cnt looks like some hours have rather high values. You’ll need to treat these outliers so that they
	don't skew the analysis and the model. 
		1. Find out the following percentiles: 10, 25, 50, 75, 90, 95, 99
		2. Decide the cutoff percentile and drop records with values higher than the cutoff. Name the new 
		   dataframe as inp2.
"""
# Outlier treatment: Finding percentiles- 10, 25, 50, 75, 90, 95, 99
space()
percentile = inp1["cnt"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.99])
print(percentile)
space()

# Outlier treatment: Decide the cutoff percentile and drop records with values higher than the cutoff. 
# Name the new dataframe as inp2.
# ANSWER: I would have chosen the 97th percentile as it is the end of the whisker.
#		  But as options were 10, 25, 50, 75, 90, 95, 99 the cutoff would be 95th percentile. (CHECK LATER)
inp2 = inp1[inp1["cnt"] < 563].copy()
inp2.reset_index(drop = True, inplace = True)
inp2 = pd.DataFrame(inp2)

# # inp2 description and details 
print(inp2.head(10))
space()
print(inp2.info())
space()
print(inp2.describe())
space()


# 7.
"""
7. Bivariate analysis
	1. Make boxplot for cnt vs. hour
		1. What kind of pattern do you see?
	2. Make boxplot for cnt vs. weekday
		1. Is there any difference in the rides by days of the week?
	3. Make boxplot for cnt vs. month
		1. Look at the median values. Any month(s) that stand out?
	4. Make boxplot for cnt vs. season
		1. Which season has the highest rides in general? Expected?
	5. Make a bar plot with the median value of cnt for each hr
		1. Does this paint a different picture from the box plot?
	6. Make a correlation matrix for variables atemp, temp, hum, and windspeed
		1. Which variables have the highest correlation?
"""

print("---------------------------BIVARIATE ANALYSIS---------------------------")
# Bivariate analysis: Make boxplot for cnt vs. hour
plt.figure(figsize = (14, 6))
sns.boxplot(x = "hr", y = "cnt", data = inp2)
plt.title("COUNT VS TIME OF THE DAY")
plt.show()
# Bivariate analysis: What kind of pattern do you see?
# ANSWER: Peak hours are 5PM - 6PM. More usage during office hours. 
#		  High usage from 7AM - 8AM and 5PM - 6PM as thats when people leave and return from work

# Bivariate analysis: Make boxplot for cnt vs. weekday
plt.figure(figsize = (14, 6))
sns.boxplot(x = "weekday", y = "cnt", data = inp2)
plt.title("COUNT VS DAY OF THE WEEK")
plt.show()
# Bivariate analysis: Is there any difference in the rides by days of the week?
# ANSWER: No. Day of the week does not affect number of rides

# Bivariate analysis: Make boxplot for cnt vs. month
plt.figure(figsize = (14, 6))
sns.boxplot(x = "mnth", y = "cnt", data = inp2)
plt.title("COUNT VS MONTH")
plt.show()
# Bivariate analysis: Look at the median values. Any month(s) that stand out?
# ANSWER: Dont know hoe median is seen in boxplots? but from plot i can say that many might not want to ride 
#		  in the summers and peek winters. But cannot be concluded

# Bivariate analysis: Make boxplot for cnt vs. season
plt.figure(figsize = (14, 6))
sns.boxplot(x = "season", y = "cnt", data = inp2)
plt.title("COUNT VS SEASON")
plt.show()
# Bivariate analysis: Which season has the highest rides in general? Expected?
# ANSWER: Fall season has the highest by a small margin. Expected to an extent.

# Bivariate analysis: Make a bar plot with the median value of cnt for each hr
hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
hour_medians = []
Group_by_hour = inp2.groupby(inp2["hr"])

for i in range(0, 24):
	median = Group_by_hour.get_group(i)["cnt"].median()
	hour_medians.append(median)

sns.barplot(x = hours, y = hour_medians)
plt.show()

# comparision between boxplot and barplot for COUNT VS HOUR
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
plt.title("Comparision btw boxplot and barplot for COUNT VS HOUR")
sns.boxplot(x = "hr", y = "cnt", data = inp2)

plt.subplot(1, 2, 2)
sns.barplot(x = hours, y = hour_medians)

plt.show()
# Bivariate analysis: Does this paint a different picture from the box plot?
# ANSWER: Similar pattern visible in both. Bar chart is easier to read and understand.

# Bivariate analysis: Make a correlation matrix for variables atemp, temp, hum, and windspeed
print(inp2[["atemp", "temp", "hum", "windspeed"]].corr())
relation = inp2[["atemp", "temp", "hum", "windspeed"]].corr()
sns.heatmap(relation, annot = True, cmap="Blues")
plt.show()
# Bivariate analysis: Which variables have the highest correlation?
# ANSWER: temp and atemp

"""
WHAT WE KNOW SO FAR
> Time of the day affects the number of rides 
	> Peak hours are 5PM - 6PM. More usage during office hours. 
	> High usage from 7AM - 8AM and 5PM - 6PM as thats when people leave and return from work
	> Similar data of rides bweteen 0-5 and 12-15
> Days of the week does not affect number of rides
> Month does not directly affect number of rides
	> Similar number of rides for months 5 to 10
> Season affects number of rides to an extent
	> Fall season has the highest number of rides by a small margin.
> temp and atemp have the highest correlation (99%)
"""


# 8.
"""
8. Data preprocessing
  A few key considerations for the preprocessing: 
  There are plenty of categorical features. Since these categorical features can’t be used in the predictive model,
  you need to convert to a suitable numerical representation. Instead of creating dozens of new dummy variables, 
  try to club levels of categorical features wherever possible. For a feature with high number of categorical 
  levels, you can club the values that are very similar in value for the target variable. 

	1. Treating mnth column
		1. For values 5,6,7,8,9,10, replace with a single value 5. This is because these have very similar values 
		   for cnt.
		2. Get dummies for the updated 6 mnth values
	2. Treating hr column
		1. Create new mapping: 0-5: 0, 11-15: 11; other values are untouched. Again, the bucketing is done in a 
		way that hr values with similar levels of cnt are treated the same.
	3. Get dummy columns for season, weathersit, weekday, mnth, and hr. You needn’t club these further as the 
	   levels seem to have different values for the median cnt, when seen from the box plots.
"""
# Data preprocessing: Treating mnth column
# Data preprocessing: replacing 0-5 to 5 as they have similar values
inp3 = inp2.copy()
inp3 = pd.DataFrame(inp3)
inp3["mnth"].replace([5, 6, 7, 8, 9, 10], 5, inplace = True)
print("month values: ", np.unique(inp3["mnth"]))
space()

# Data preprocessing: getting dummy values for the updated 6 mnth values (will be done later)

# Data preprocessing: Treating hr column
# Data preprocessing: replacing 0-5 as 0, 11-15 as 11. This is done as they have similar values.
print("hour values: ", np.unique(inp3["hr"]))
inp3["hr"].replace([0, 1, 2, 3, 4, 5], 0, inplace = True)
print("hour values: ", np.unique(inp3["hr"]))
inp3["hr"].replace([11, 12, 13, 14, 15], 11, inplace = True)
print("hour values: ", np.unique(inp3["hr"]))
space()

# Data preprocessing: getting dummy values for the updated hr values (will be done later)

# Data preprocessing: Get dummy columns for season, weathersit, weekday, mnth, and hr

requires_dummy_value = ["season", "weathersit", "weekday", "mnth", "hr"]
inp3 = pd.get_dummies(inp3, columns = requires_dummy_value)
print(inp3.columns)
space()
print(inp3.info())


# 9.
"""
9. Train test split: Apply 70-30 split.
	- call the new dataframes df_train and df_test
"""
# Train test split: Apply 70-30 split.
# Train test split: call the new dataframes df_train and df_test
df_train, df_test = train_test_split(inp3, test_size = 0.3, random_state = 42)
print(df_train.shape[0], df_test.shape[0])
space()
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)


# 10.
"""
10. Separate X and Y for df_train and df_test. For example, you should have X_train, y_train from df_train. 
	y_train should be the cnt column from inp3 and X_train should be all other columns.
"""
# get x_train and y_train from df_train
y_train = df_train["cnt"].copy()
x_train = df_train.drop("cnt", axis = 1)
print(y_train)
space()
print(x_train)
space()

# get x_test and y_test from df_test
y_test = df_test["cnt"].copy()
x_test = df_test.drop("cnt", axis = 1)
print(y_test)
space()
print(x_test)
space()


# 11.
"""
11. Model building
	> Use linear regression as the technique
	> Report the R2 on the train set
"""
# Model building: Use linear regression as the technique
lr1 = LinearRegression()
lr1.fit(x_train,y_train)
Yhat1 = lr1.predict(x_train)

# Model building: Report the R2 on the train set
R_squared_1 =r2_score(y_train, Yhat1)
print("R- squared value of training dataset: ", R_squared_1)


# 12.
"""
12. Make predictions on test set and report R2.
"""
# Model building: Use linear regression as the technique
lr2 = LinearRegression()
lr2.fit(x_test,y_test)
Yhat2 = lr1.predict(x_test)

# Model building: Report the R2 on the train set
R_squared_2 =r2_score(y_test, Yhat2)
print("R- squared value of Test dataset: ", R_squared_2)