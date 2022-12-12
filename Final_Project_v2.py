#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


# In[2]:


s = pd.read_csv('./social_media_usage.csv')


# In[3]:


s.head()


# In[ ]:


#2. Make a function using np.where


# In[4]:


def clean_sm(x): 
    x = np.where(x == 1,
                1,
                0)
    return x


# In[5]:


clean_sm(1)


# ***

# In[ ]:


#2. Test the function on a data frame


# In[6]:


data = [['tom', 10], ['nick', 15], ['john', 12]]


# In[7]:


toy = pd.DataFrame(data, columns=['Name', 'Age'])


# In[8]:


# In[9]:


toydf = clean_sm(toy)


# In[10]:


print(toydf)


# In[ ]:





# ***

# 3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[11]:


ss = pd.DataFrame({
    "income": np.where(s["income"] <= 9, s["income"], np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"] <= 2, s["par"], np.nan),
    "married": np.where(s["marital"] <= 6, s["marital"], np.nan),
    "female": np.where(s["gender"] <= 2, s["gender"], np.nan),
    "age": np.where(s["age"] <= 97, s["age"], np.nan),
    "sm_li": clean_sm(s["web1h"])})


# In[12]:


ss = ss.dropna()


# In[13]:


ss.head(50)


# #### 4. Create Target Vector (y) and feature set (x)

# In[15]:


y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]


# #### imort the regression models

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# #### 5. Split data into training and test sets

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 987)


# - Explain: In this section we are splitting our data up into a training section and a testing section. By having a training section, we allow our algorithim to learn the data. We then apply the model to the test set. We have held out 80% of the data
#     to be used as our training data. The model will use the remaining 20% to predict.

# #### 6. Initialize algorithm

# In[18]:


lr = LogisticRegression()


# In[19]:


lr.fit(X_train, y_train)


# #### 7. Evaluate Model

# In[20]:


y_pred = lr.predict(X_test)


# #### 8. Confusion Matrix

# In[21]:


#Confustion Matrix - use to calc by hand

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns = ["Predicted negative", "Predicted positive"],
            index = ["Actual negative", "Actual Positive"]).style.background_gradient(cmap = "PiYG")


# In[22]:


print(classification_report(y_test, y_pred))








new_pred = pd.DataFrame({
    "income": [8],
    "education": [7],
    "parent": [2],
    "married": [1],
    "female": [2],
    "age": [42],
    })


# In[27]:


new_pred["prediction_linkedin"] = lr.predict(new_pred)


# In[28]:





# In[29]:


user = [8, 7, 2, 1, 2, 42]


# In[30]:


predicted_class = lr.predict([user])


# In[31]:


prob = lr.predict_proba([user])


# In[29]:


print(prob[0][1])


# In[32]:


user2 = [8, 7, 2, 1, 2, 82]


# In[33]:


predicted_class2 = lr.predict([user2])


# In[34]:


prob2 = lr.predict_proba([user2])


# In[35]:


print(prob2[0][1])


# - When changing the age of the predicted user, while holding all other variables constant, we see the probability of 
#     a person being a LinkedIn user drop to 31% from 59%


st.image('images.png')
st.markdown("# LinkedIn User Prediction Application")
user2 = [8, 7, 2, 1, 2, 82]
income = st.selectbox(
    "What's your Income?",
    ("Less than $10k",
    "$10k - $20k",
    "$20k - $30k",
    "$30k - $40k",
    "$40k - $50k",
    "$50k - $75k",
    "$75k - $100k",
    "$100k - $150k",
    "$150k +")
)

if income == "Less than $10k": income_value = 1
elif income == "$10k - $20k": income_value = 2
elif income == "$20k - $30k": income_value = 3
elif income == "$30k - $40k": income_value = 4
elif income == "$40k - $50k": income_value = 5
elif income == "$50k - $75k": income_value = 6
elif income == "$75k - $100k": income_value = 7
elif income == "$100k - $150k": income_value = 8
elif income == "$150k +": income_value = 9

education = st.selectbox(
    "What's your level of Education?",
    ("Less than high school (Grades 1 - 8 or no formal education",
    "Highschool - incomplete",
    "High school graduate (includes diploma or GED)",
    "Some college but no degree (includes community college)",
    "Two-year associate degree (from college or university)",
    "Four-year bachelors degree",
    "Some postgraduate or professional schooling, no postgraduate degree",
    "Postgraduate or professional degree (including Masters, Doctorate, medical or law degree"
    )
)

if education == "Less than high school (Grades 1 - 8 or no formal education": education_value = 1
elif education == "Highschool - incomplete": education_value = 2
elif education == "High school graduate (includes diploma or GED)": education_value = 3
elif education == "Some college but no degree (includes community college)": education_value = 4
elif education == "Two-year associate degree (from college or university)": education_value = 5
elif education == "Four-year bachelors degree": education_value = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree": education_value = 7
elif education == "Postgraduate or professional degree (including Masters, Doctorate, medical or law degree": education_value = 8

parent = st.selectbox(
    "Are you currently a parent of a child under 18 living in your home?",
    ("Yes",
    "No")
)

if parent =="Yes": parent_value = 1 
elif parent == "No": parent_value = 2

married = st.selectbox(
    "Marital Status",
    ("Married",
    "Living with partner",
    "Divorced",
    "Separated",
    "Widowed",
    "Never been married")
)
if married == "Married": married_value = 1
elif married == "Living with partner": married_value = 2
elif married == "Divorced": married_value = 3
elif married == "Separated": married_value = 4
elif married == "Widowed": married_value = 5
elif married == "Never been married": married_value = 6

gender = st.selectbox(
    "Please select your Gender",
    ("Male",
    "Female",
    "Other")
)

if gender == "Male": gender_value = 1
elif gender == "Female": gender_value = 2
elif gender == "Other": gender_value = 3

age_value = st.number_input("What's your age?", min_value = 0, max_value = 98)

user3 = [income_value, education_value, parent_value, married_value, gender_value, age_value]
predicted_class3 = lr.predict([user3])
prob3 = lr.predict_proba([user3])


if predicted_class3 == 0: predicted_value = "Non User"
elif predicted_class3 == 1: predicted_value = "LinkedIn User"

button_click = st.button("Click here to find the probability that you are a LinkedIn User")

if button_click == True: st.markdown(f"The probability of being a LinkedIn user: {prob3[0][1]}")
if button_click == True: st.markdown(f"Prediction: {predicted_class3[0]} - {predicted_value}")