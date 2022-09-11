import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for visualzation
import matplotlib.pyplot as plt # for visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import streamlit as st

# Use the full page instead of a narrow central column
st.set_page_config()

# use css property
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

c0 = st.container()
c1 = st.container()
c2 = st.container()
c3 = st.container()
c4 = st.container()
c5 = st.container()
c6 = st.container()

with c0:
    st.markdown("<h1 style='text-align: center; color: blue;'>BANGALURU HOUSE PRICE PREDICTION</h1>", unsafe_allow_html=True)
    #st.image("Banglore.jpg")


# Reading the data
# Creating two separate variable to check how many duplicate rows we have
data_ini = pd.read_csv('BHP.csv')
data = data_ini.drop_duplicates()

# Checking the information
print('Original data             : ', data_ini.shape)
print('After removing duplicates : ', data.shape)


with c1:
    st.write(data.head())

# * Notice that, the column 'total_sqft' which contains measurement information in sqft has 'object' as data type which we need to change to float/int

# ## 'total_sqft' column :
# 1. The column is in object form instead of numeric form
# 2. Some of the rows contain different measuring units other than sqft (mentioned in the column name). Those other measurements are in Sq yard, Acre, Perch, Guntha etc. So, we need to convert them in SqFt unit
# 3. Some of the values are in range. for example, 1020 - 1130, 1133 - 1384 etc. We need to make them to a single quantity so that we can group them as per our requirements

# Making the measurement unit unique

# This function extract the string
def finding_string(a):
    temp = ''
    for i in a:
        if (i.isalpha()) | (i == "."):
            temp = temp + i
    return temp

# This function extract the number
def finding_number(a):
    temp = ''
    for i in a:
        if (i.isdigit()) | (i == " ") | (i == "-") | (i == "."):
            temp = temp + i
    return temp

# Creating two columns : 'Area0' and 'Unit'
# 'Area0' column contains numeric values 
# 'Unit' column contains measurement units
data['Area0'] = data['total_sqft'].apply(finding_number)
data['Unit']  = data['total_sqft'].apply(finding_string)

# *  We still need to take care of range values. so, from range values we are considering the maximum value. 
# *  for example, from 1020 - 1130, we are considering 1130

# 'total_sqrt' column has data in range format and we are considering the maximum number from the range
data['Area'] = np.where(data['Area0'].str.contains(" - "), data['Area0'].apply(lambda x : x[x.rfind(" "):]), data['Area0'])
data['Area'] = data['Area'].str.strip()
data['Area'] = np.where(data['Area'].str.endswith("."), data['Area'].str.replace(".", ""), data['Area'])
data['Area'] = np.round(data['Area'].astype(float), 0).astype(int)

# * We will now work on Unit column
data['Unit'] = np.where(data['Unit'].str.startswith("."), data['Unit'].str.replace(".", ""), data['Unit'])
data['Unit'].unique()

# * There are 7 types of unit other than SqFt. We need to apply appropriate conversion quantities and apply them to get all the units in SqFt format.
# Converting different units (other than sqft) into sqft
cond1 = (data['Unit'] == 'SqMeter') | (data['Unit'] == 'Sq.Meter')
cond2 = (data['Unit'] == 'Perch')
cond3 = (data['Unit'] == 'SqYards') | (data['Unit'] == 'Sq.Yards')
cond4 = (data['Unit'] == 'Acres')
cond5 = (data['Unit'] == 'Cents')
cond6 = (data['Unit'] == 'Guntha')
cond7 = (data['Unit'] == 'Grounds')

cond_values = [data['Area'] * 10.7639104, data['Area'] * 272.25, data['Area'] * 9, data['Area'] * 43560, data['Area'] * 435.6, data['Area'] * 1089, data['Area'] * 2400]

data['Area_sqft'] = np.select([cond1, cond2, cond3, cond4, cond5, cond6, cond7], cond_values, data['Area'])

data['Area_sqft'] = np.round(data['Area_sqft'], 0).astype(int)
data['Area_sqft'].sum()

data = data[data['Area_sqft'] >= 350]

# ## 'size' column :
# * Let us start with size - we will replace the word 'Bedroom' with 'BHK'
data['size'] = data['size'].str.replace("Bedroom", "BHK")
print(data[data['size'].isnull()][['area_type']].value_counts())
data = data[~data['size'].isnull()]
data['size'].value_counts()

# * We can remove the rows having only one observation per size
data = data[(data['size'] != '11 BHK') & (data['size'] != '27 BHK') & (data['size'] != '19 BHK') & (data['size'] != '16 BHK') & (data['size'] != '43 BHK') & 
            (data['size'] != '14 BHK') & (data['size'] != '12 BHK') & (data['size'] != '13 BHK') & (data['size'] != '18 BHK')]

# ## Let us check for 'NULL' values
# Checking the percentage of null values in a column
print(data.isnull().sum()/len(data) * 100)

# ## 'society' column :
# * ##### It is clear that the column 'society' will not be that helpful in the price prediction since it contains 41% of null values. So, we will drop it
# Droping 'society' column
del (data['society'])
data = data.drop_duplicates()
print(data.columns)

# 'Balcony' and 'Bath' column :
# * We need to fill na values in 'balcony' and 'bath' columns
# * We will fill them with mode value
data['balcony'] = data['balcony'].fillna(data['balcony'].mode()[0])
data['bath'] = data['bath'].fillna(data['bath'].mode()[0])
data.isnull().sum()

# ## 'Location' column :
# we will only consider those locations which have 5 or more data points
location_count = data['location'].value_counts().reset_index()
location_count_5 = location_count[location_count['location'] >= 5]
data['location_flag'] = np.where(data['location'].isin(location_count_5['index']), 0, 1)
data = data[data['location_flag'] == 0]


# ## Let us check for 'OUTLIERS' :
fig, axes = plt.subplots(figsize = (25,8), nrows = 1, ncols = 2)
sns.boxplot(data = data, y = 'Area_sqft', x = 'size', ax = axes[0])
sns.boxplot(data = data, y = 'price', x = 'size', ax = axes[1])
plt.show()

# * 'price' shows some outliers at different size levels
# * Here, the correct way of removing outliers would be checking the outliers at 'size x total_sqft' level and not at 'total_sqft' level because if we only consider 'total_sqft' then it might possible that because of the huge variation in prices, large houses (8BHK, 9BHK) will get flagged as outliers.
# Creating a function for IQR
def IQR(data, column):
    q1 = dict(data.loc[:, column].quantile([0.25]))[0.25]
    q2 = dict(data.loc[:, column].quantile([0.50]))[0.50]
    q3 = dict(data.loc[:, column].quantile([0.75]))[0.75]
    
    iqr = q3 - q1
    lb = q1 - (1.5 * iqr)
    ub = q3 + (1.5 * iqr)
    
    data['IQR_FLAG'] = np.where((data[column] < lb) | (data[column] > ub), 1, 0)
    return (data['IQR_FLAG'].sum())

# Applying Outlier detection method IQR on each pair of 'size x Area_sqft'
temp1 = pd.DataFrame()
for temp_size in data['size'].unique():
    temp_data1 = data[data['size'] == temp_size]
    temp2 = pd.DataFrame()
    for temp_sqft in temp_data1['Area_sqft'].unique():
        temp_data2 = temp_data1[temp_data1['Area_sqft'] == temp_sqft]
        IQR(temp_data2, 'price')
        temp2 = pd.concat([temp2, temp_data2])
    temp1 = pd.concat([temp1, temp2])

# Removing the outliers
data = temp1[temp1['IQR_FLAG'] == 0]

fig, axes = plt.subplots(figsize = (25,8), nrows = 1, ncols = 2)
sns.boxplot(data = data, y = 'Area_sqft', x = 'size', ax = axes[0])
sns.boxplot(data = data, y = 'price', x = 'size', ax = axes[1])
plt.show()

# Getting information after removing the outliers
data.describe()

# * Check for Bath column - It shows that we have records which hav 14 bathrooms
# * It is unrealistic to have bathrooms more than number of rooms. For example, 3BHK house can have 3 or 4 bathrooms (1 for each room and 1 for common use)
# Check how many bathrooms each size has
plt.figure(figsize = (15, 8))
sns.scatterplot(data = data, x = 'size', y = 'bath')
plt.show()

# Removing the outliers from 'Bath' column -  we are allowing upto the same number of bathrooms for each size 
def split1(a):
    return int(a[0])
data['bath_flag'] = np.where(data['bath'] <= data['size'].apply(split1), 0, 1)

# Check how the number of bathrooms has changed
plt.figure(figsize = (15, 8))
sns.scatterplot(data = data[data['bath_flag'] == 0], x = 'size', y = 'bath')
plt.show()
data = data[data['bath_flag'] == 0]
data.describe()

# Removing null value row from loacation
data = data[~data['location'].isnull()]

# Applying LabelEncoding and OneHotEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['size'] = le.fit_transform(data['size']) 
data['size'].unique()

location = pd.Series(data['location'].sort_values().unique())
sqft     = pd.Series(data['Area_sqft'].sort_values().unique())
bhk      = pd.Series(data['size'].sort_values().unique())
bath     = pd.Series(data['bath'].sort_values().unique())

data     = pd.get_dummies(data, columns = ['location'], drop_first = True)

# Droping the unnecessary columns
data.drop(['area_type', 'availability', 'total_sqft', 'Area0', 'Unit', 'Area', 'location_flag', 'IQR_FLAG', 'bath_flag'], axis = 1, inplace = True)

# Reindexing 'price' column to the end of the dataframe
data = data.reindex(columns = [col for col in data.columns if col != 'price'] + ['price'])
data1 = data.drop_duplicates()

# Splitting data
X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

print('train : ', X_train.shape)
print('test  : ', X_test.shape)
print('data  : ', data1.shape)

# # Lets try some models
# Model 1 - Linear Regression
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Model 2 - Random Forest Regressor
model2 = RandomForestRegressor(n_estimators = 50, random_state = 100)
model2.fit(X, y)

# Model 3 - K-Nearest Regressor
model3 = KNeighborsRegressor(n_neighbors = 10)
model3.fit(X_train, y_train)

# # Accuracy of models
# Model 1 - Linear Regression
print ("Linear Model accuracy on train data : ", r2_score(model1.predict(X_train), y_train) * 100)
print ("Linear Model accuracy on test data  : ", r2_score(model1.predict(X_test), y_test) * 100)

# Model 2 - Random Forest Regressor
print ("RF Regressor accuracy on train data : ", r2_score(model2.predict(X_train), y_train) * 100)
print ("RF Regressor accuracy on test data  : ", r2_score(model2.predict(X_test), y_test) * 100)

# Model 3 - K-Nearest Regressor
print ("KN Regressor accuracy on train data : ", r2_score(model3.predict(X_train), y_train) * 100)
print ("KN Regressor accuracy on test data  : ", r2_score(model3.predict(X_test), y_test) * 100)

# To predict the price
def predict_price(Location, sqft, bhk, bath):
    loc_index = np.where(data1.columns == 'location_' + Location)[0][0]
    x = np.zeros(len(data1.columns) - 1)
    x[3] = sqft
    x[1] = bath
    x[0] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return ' {}   LAKHS'.format (np.round(model2.predict([x])[0], 2) )


with c1:
    st.markdown("<h2 style='text-align: center; color: black;'>We have applied 3 different models</h2>", unsafe_allow_html=True)

with c2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style = 'text-align : right; color : black; '>LINEAR REGRESSION </h4>", unsafe_allow_html = True)
        st.write("Accuracy on Train Data : ")
        st.write(r2_score(model1.predict(X_train), y_train) * 100)
        st.write("Accuracy on Test Data : ")
        st.write(r2_score(model1.predict(X_test), y_test) * 100)

    with col2:
        fig, ax = plt.subplots()
        plt.scatter(model1.predict(X_train), y_train)
        plt.title('LINEAR REGRESSION')
        st.pyplot(fig)

with c3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style = 'text-align : right; color : black; '>RANDOM FOREST REGRESSOR</h4>", unsafe_allow_html = True)
        st.write("Accuracy on Train Data : ")
        st.write(r2_score(model2.predict(X_train), y_train) * 100)
        st.write("Accuracy on Test Data : ")
        st.write(r2_score(model2.predict(X_test), y_test) * 100)

    with col2:
        fig, ax = plt.subplots()
        plt.scatter(model2.predict(X_train), y_train)
        plt.title('RANDOM FOREST REGRESSOR')
        plt.xlabel("ACTUAL")
        plt.ylabel('PREDICTED')
        st.pyplot(fig)

with c4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style = 'text-align : right; color : black; '>KNEAREST REGRESSOR</h4>", unsafe_allow_html = True)
        st.write("Accuracy on Train Data : ")
        st.write(r2_score(model3.predict(X_train), y_train) * 100)
        st.write("Accuracy on Test Data : ")
        st.write(r2_score(model3.predict(X_test), y_test) * 100)

    with col2:
        fig, ax = plt.subplots()
        plt.scatter(model3.predict(X_train), y_train)
        plt.title('KNEAREST REGRESSOR')
        plt.xlabel("ACTUAL")
        plt.ylabel('PREDICTED')
        st.pyplot(fig)

with c5:
    st.markdown("<h2 style = 'text-align : center; color : black; '>-:   CONCLUSION   :-</h2>", unsafe_allow_html = True)
    st.markdown("<h4 style = 'text-align : left; color : black; '>Random Forest gives better accuracy (approx. 92.41%) than Linear Regression and KNearestRegressor</h4>", unsafe_allow_html = True)

with c6:
    st.markdown("<h2 style = 'text-align : center; color : black; '>LET US TEST THE MODEL</h2>", unsafe_allow_html = True)
    col1, col2 = st.columns(2)
    with col1:
        with st.form('Input_Form'):
            Location = st.selectbox("Please select the location : ", options = location, index = 0)
            sqft     = st.selectbox("Please select the Area size (in sqft) : ", sqft)
            bhk      = st.selectbox("Pleaes select the plan : ", bhk)
            bath     = st.selectbox("Please select number of bathrooms : ", bath)
            button1 = st.form_submit_button("Predict")

    with col2:
        st.markdown("<h3 style = 'text-align : center; color : black; '>Your desired property costs </h3>", unsafe_allow_html = True)
        if button1:
            html_str = f"""
            <style>
            p.a {{
                font-type : bold;
                font-size : 40px;
                font-family : Courier New;
                color : Blue;
                text-align : center;
            }}
            </style>
            <p class="a">{predict_price(Location, sqft, bhk, bath)}</p>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            