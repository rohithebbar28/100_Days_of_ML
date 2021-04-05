import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("AB_NYC_2019.csv")
#print(data.head())
print(data.columns)
print(data.describe())
"""
This will show you the count, mean, std, min, 25% 50%, 75%, and a max of each column in the table.
Next, we need to see how many non-null counts in each column. If the number of non-null values equal to the number
of the rows on the table, there are no null values that need to be taken care of in that column. However, be aware 
that the column may still contain some nonsense values or values that are not supposed to be in it.
"""

print(data.info())
# As we can see that last_review and reviews_per_month has some null values.

print(data.isnull().sum())

# Now i will drop the duplicate values
data.duplicated().sum
data.drop_duplicates(inplace=True)

"""
Then, in order to replace null values, I replace them with values that are appropriate for each column. 
My goal is to make sure that no column contains no null values.
"""

data.fillna({'reviews_per_month':0}, inplace=True)
data.fillna({'name':"No name"}, inplace=True)
data.fillna({'host_name':"No Host Name"}, inplace=True)
data.fillna({'Last_review':"No Review"}, inplace=True)

# Data Visualisation
# Which neighbourhood has the most Airbnb?

data['neighbourhood_group'].value_counts().sort_index().plot.barh()


# From the bar graph we can see that Brookyln and Manhattan were the most Airbnb bookings

"""
The next step is to get the correlation between different values in the table. The goal is to see which feature 
variables will be important in determining the price of New York Airbnb.
"""

corr = data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
data.columns




corr_1 = data.corr(method='pearson')
plt.figure(figsize=(10,5))
sns.heatmap(corr_1, annot=True)
data.columns
#plt.show()

# More Visualisations
plt.figure(figsize=(10,7))
sns.scatterplot(data.longitude,data.latitude, hue=data.availability_365)
plt.ioff()


# Linear Regression Model

from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

"""
After that, we need to encode columns that contain non-numerical values. The reason is that machine learning
models are not capable of interpreting words, a.k.a non-numerical values.
"""
le = LabelEncoder()
data['neighbourhood_group'] = le.fit(data['neighbourhood_group'])

data['neighbourhood'] = le.fit(data['neighbourhood'])

data['room_type'] = le.fit(data['room_type'])

x = data.iloc[:,[0,7]]
y = data['price']

# Spliting it into train and test set

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=353)

# instantiate fit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
print(f"Accuracy on test set : {linreg.score(X_test, y_test):.3f}")

"""
My accuracy score was around 0.030, which is 3% and this might imply that either the metric is not suitable for 
this data set or the data is not impressive enough.
Now, we can calculate our predictions after training the model using a simple linear regression model.

"""
predictions = linreg.predict(X_test)
error = pd.DataFrame(np.array(y_test).flatten(), columns=['actual'])
error['prediction'] = np.array(predictions)
print(error.head(10))












