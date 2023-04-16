#!/usr/bin/env python
# coding: utf-8

# <h1><center> Aerofit </center></h1>

# ## Descriptive Statistics & Probability

# ## `1. Defining the Problem Statement:`
# 1. To identify the characteristics of the target audience for each type of treadmill offered by Aerofit.
# 2. To provide a better recommendation of the treadmills to the New Customers.
# 3. To investigate whether there are differences across the product with respect to customer characteristics

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[3]:


aerofit = pd.read_csv("C:/Users/Rajeshri Jogi/Desktop/Data/Project/3. Aerofit/aerofit_treadmill.txt")


# In[4]:


aerofit


# ## `Dataset `
# The company collected the data on individuals who purchased a treadmill from the AeroFit stores during the prior three months. The dataset has the following features:
# 
# 1. **Product Purchased**:	KP281, KP481, or KP781
# 2. **Age**:	                In years
# 3. **Gender**:	            Male/Female
# 4. **Education**:	        In years
# 5. **Marital Status**:	    Single or partnered
# 6. **Usage**:	            The average number of times the customer plans to use the treadmill each week.
# 7. **Income**:	            Annual income (in USD)
# 8. **Fitness**:	            Self-rated fitness on a 1-to-5 scale, where 1 is the poor shape and 5 is the excellent shape.
# 9. **Miles**:	            The average number of miles the customer expects to walk/run each week

# ## `1.2. Becoming Aquainted with Data`

# ### `Analysis:`
# 1. The data has records for total 180 customers.
# 2. There is no null/NaN values in the data.
# 3. Total no. of **columns = 9 and rows = 180.**
# 4. Customers Age lays between 18 - 50, which I have categorized in 9 bins (equally) as **Age_group.**
# 5. Customers Income lays between USD 29000 - USD 105000, which I categorized in 5 bins(equally) as **Grade.**

# In[5]:


aerofit.info()


# In[6]:


aerofit.shape


# In[7]:


aerofit.isnull().sum()


# In[9]:


df = aerofit.copy(deep = True)
df


# In[10]:


df.loc[df.duplicated()]


# In[214]:


bins = [29000,44000,59000,74000,99000,105000]
labels = ["E","D","C","B","A"]
df["Grade"] = pd.cut(x = df["Income"],bins = bins,labels = labels)
df


# In[208]:


bins = [17,22,26,30,34,38,42,46,50]
labels = ["17-21","22-25","26-29","30-33","34-37","38-41","42-45","46-50"]
df["Age_group"] = pd.cut(x = df["Age"],bins = bins,labels = labels)
df


# ## `2. Non-Graphical Analysis: Value counts and Unique Attributes`

# ### `Analysis:`
# 1. 60% of the customers are **Married.**
# 2. 58% of the customers are **Male.**
# 3. Highest product purchased is **KP281.**
# 4. Aveage Income of customers is approx **USD 54000**, Average Age of the customer is **28yrs.**

# In[153]:


df.describe(include=object)


# In[186]:


df.describe()


# ## `3. Visual Analysis - Univariate & Bivariate`

# ### `Analysis and Insights:`
# 1. Most purchased threadmill is **KP281.**
# 2. Highest threadmills are purchased by customers of **age 25yrs.**
# 3. Median age of customers buying threadmill(any) is **26yrs.**
# 4. There are very few customers **above 45yrs**, who purchased threadmills.
# 5. **KP781** is bought by only customer who's income is **above USD 60000.**
# 6. Average usage by customers of threadmill **KP281 and KP481** is *3 days a week*, however, **KP781** avg usage is *4 days a week.*
# 7. Most of customers purchasing **KP281 & KP481** lays in the Income group E and D which is between **USD 29k - 58k**, 
#    while **KP781** income group is B the most i.e **USD 74k-98k.**
# 8. Most of the Threadmills are purchased by customers age between **18-37yrs** and Income beteen **USD 30k-55k.**
# 9. **Outliers**: Female above 45yrs purchasing **KP281**, Male above 40yrs purchasing **KP781**.
# 10. Customers who rated themselves on Fitness as 3 are more likely to purchase **KP281 or KP481** and rated 4 or 5 are likely to     purchase **KP781**.
# 11. Female with usage 2-3days in a week are likely to purchase **KP281**, Male 3-4days in a week are likely to purchase             **KP281**

# In[13]:


sns.countplot(data = df,
             x = "Product")


# In[135]:


plt.figure(figsize=(8,4))
sns.countplot(data = df,
             x = "Age")


# In[116]:


sns.boxplot(data = df,
           x = "Product",
           y = "Age")


# In[117]:


sns.boxplot(data = df,
           x = "Product",
           y = "Income")


# In[213]:


sns.countplot(data = df,
           hue = "Product",
           x = "Usage")


# In[168]:


plt.figure(figsize=(8,5))
sns.countplot(data = df,
               x = "Grade",
               hue = "Product")


# In[188]:


plt.figure(figsize=(10,6))
sns.scatterplot(data = df,
               x = "Income",
               y = "Age",
               hue = "Product")


# In[190]:


plt.figure(figsize=(10,6))
sns.boxplot(data = df,
               x = "Gender",
               y = "Age",
               hue = "Product")


# In[215]:


plt.figure(figsize=(12,6))
sns.countplot(data = df,
               x = "Fitness",
               hue = "Product")


# In[127]:


plt.figure(figsize=(10,5))
sns.boxplot(data = df,
               x = "Product",
               y = "Usage",
               hue = "Gender")


# ## `4. Correlation between among all the factors related to Product purchase`

# ### `Insights:`
# 1. Fitness,Miles and Usage are **higly related.**
# 2. Income is **moderately related.**
# 3. Education is **least related** to any factors of the customers purchase.

# In[144]:


plt.figure(figsize=(8,5))
sns.heatmap(df.corr(),annot = True)


# ## `5. Marginal and Conditional Probability`

# ### `Analysis and Insights:`
# 1. Given that customer is **Female**,the probability that she will buy KP281 is **52%**,KP481 is **38%** & KP781 is **10%**
# 2. Given that customer is **Male**,the probability that he will buy KP281 is **38%**, KP481 is **30%** & KP781 is **32%**
# 3. Given that the usage is **3 days a week**, the probability that he/she will buy KP281 is **54%**, KP481 is **45%** and KP781 is **1%**.
# 3. Given that the usage is **4 days a week**, the probability that he/she will buy KP281 is **42%**, KP481 is **23%** and KP781 is **35%**
# 4. Given that the usage is **5 days a week**, the probability that he/she will buy KP281 is **12%**, KP481 is **18%** and KP781 is **70%**
# 5. Given that a Female is Married and her usage of threadmill is 3days a week, the probability that he/she will buy KP281 is **52%**.
# 6. Given that the Income of a Customer is between USD 44k-58k, the probability that he/she purchase a threadmill is **50%**.
# 7. Given that the Income of a Customer is between USD 29k-43k, the probability that he/she purchase **KP281** is **67%**.
# 8. Given that the Income of a Customer is between USD 74k-105k, the probability that he/she purchase **KP781** is **100%**
# 9. Given that the Income of a Customer is between USD 44k-58k and is in age between 22-25yrs, the probability that he/she purchase **KPI481** is **68%**.
# 10. Given that the Income of a Customer is between USD 74k-98k and is in age between 26-29yrs, the probability that he/she purchase **KPI781** is **72%**.

# In[209]:


pd.crosstab(df["Product"],df["Gender"],margins = True, normalize = True)


# In[217]:


pd.crosstab(df["Product"],df["Usage"],margins = True, normalize = True)


# In[222]:


pd.crosstab(df["Product"],df["MaritalStatus"],margins = True, normalize = True)


# In[194]:


pd.crosstab([df["Product"],df["Gender"],df["MaritalStatus"]],df["Usage"],margins = True,normalize = True)


# In[196]:


pd.crosstab(df["Product"],df["Grade"],margins = True)


# In[220]:


pd.crosstab([df["Product"],df["Grade"]],df["Age_group"],margins = True)


# In[221]:


pd.crosstab(df["Product"],df["Age_group"],margins = True)


# In[223]:


pd.crosstab(df["Product"],df["Usage"],margins = True)


# ## `6. Customer Profile for Each Aerofit threadmill Product:`

# ![Capture.PNG](attachment:Capture.PNG)

# ## `7. Recommendations:`
# 1. There is hardly any difference in price of KP281 and KP481, a bit increase in price and features of KP481 will diversify the customers and more customers would tend to buy **KP481.**
# 2. The median income and average income is between USD 50k - 54k, it is more likely that the customers would spend more if there some new features in **KP481**.
# 3. Target customers between **22-25yrs** and create more champaigns around this age group.
# 4. A **female athelete or a blogger** promoting KP481 with new features will attract more customers.
# 5. The whole customer demography lays between **20-29yrs of customers and Partnered customers**, keeping this in mind new features can be added, campaigns can be ran accordingly.
# 6. **A brand ambassador in mid 20's** would help Aerofit reach greater Audience.
# 7. As the data says most the customers are partenered, adding a new line to promotion like **"make the best use of it with your partner"**
