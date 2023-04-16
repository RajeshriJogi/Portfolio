#!/usr/bin/env python
# coding: utf-8

# <h1><center> Yulu - Hypothesis Testing </center></h1>

# ##  <span style='background :yellow' > Business Problem and Analyzing Basic Materics: </span>
# 1. Understand the **factors affecting the demand** for these shared electric cycles in the Indian market.
# 2. Which **variables are significant** in predicting the demand for shared electric cycles in the Indian market?
# 3. How well those **variables describe** the electric cycle demands

# In[165]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as st


# In[166]:


#imported data
yulu = pd.read_csv("C:/Users/Rajeshri Jogi/Desktop/Data Science/PROJECT/5. Yulu/bike_sharing.csv")


# In[167]:


yulu.head()


# In[168]:


yulu.shape


# In[169]:


yulu.info()


# In[170]:


yulu.describe()


# In[171]:


yulu.isnull().sum()


# # <span style='background :yellow' > Exploratory Data Analysis: </span>

# ### `Insights:`
# 1. **66%** Irrespective of Season the most cycles are rented on weather **Category 1 i.e. Clear, Few clouds, partly cloudy, partly cloudy**
# 2. 68% data cosists of workingdays and 32% is a holiday.
# 3. Highest cycles are rented at **5pm > 6pm > 8am**
# 4. Most of cycles rented count is betweem **0 - 200**.
# 5. Average temperature in the data is 20.
# 6. **Weather Category 4** (Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog) is a **Outlier.**
# 7. **In Weather 1 (Clear, Few clouds, partly cloudy, partly cloudy) and in Season 3 (Fall)** most of the cycles are rented.
# 8. There is hardly any difference in cycles rented on holidays vs workingdays.
# 9. **Highest correlation is seen between Registered and Count > Casual and Count.**

# In[172]:


df = yulu.copy(deep=True)


# In[173]:


# separating date and time for further analysis
df['date'] = pd.to_datetime(df['datetime']).dt.date
df['time'] = pd.to_datetime(df['datetime']).dt.hour
df.drop(["datetime"],axis = 1, inplace =True)
df


# In[174]:


# count of data as per the year
pd.to_datetime(df['date']).dt.year.value_counts()


# In[175]:


# count of data as per the month
pd.to_datetime(df['date']).dt.month.value_counts()


# In[176]:


df.info()


# In[177]:


# workingday column: working day = 1, holiday = 0
# holiday column: national holiday=1, not a national holiday: 0
pd.crosstab(df["holiday"],df["workingday"],margins = True)


# In[232]:


# comparison of weather and season on cycle demand
pd.crosstab(df["season"],df["weather"],margins = True)


# In[254]:


sns.histplot(data = df, x= "count",bins=50)


# In[213]:


sns.histplot(data = df, x = "temp", bins = 10)


# In[226]:


# changing the data type to category, for EDA
df2 = yulu.copy(deep=True)
df2["weather"] = df2["weather"].astype('category')
df2["season"] = df2["season"].astype('category')
df2["workingday"] = df2["workingday"].astype('category')
df2["holiday"] = df2["holiday"].astype('category')


# In[236]:


sns.boxplot(data = df2, x = "weather", y = "count")


# In[237]:


sns.boxplot(data = df2, x = "season", y = "count")


# In[238]:


sns.boxplot(data = df2, x= "workingday", y = "count")


# In[ ]:





# In[180]:


# count of cycles rented at which hour
plt.figure(figsize=(10,6))
sns.scatterplot(data=df,
             x = "time",
                y = "count")


# In[230]:


plt.figure(figsize=(10,6))
sns.heatmap(df2.corr(),annot=True)


# # <span style='background :yellow' > Hypothesis Testing:</span>

# ## `1. Working Day has an effect on the number of electric cycles rented.`

# ### `Insights and Proof:`
# 1. **Hypothesis formulation:** 
#     **H0: Avg cycles rented on workingday = Avg cycles rented on a holiday
#     H1: Avg cycles rented on workingday != Avg cycles rented on a holiday**
# 2. **Test : 2 sample T-Test** as these groups are independent and have equal variances
# 3. **Result:** Ttest_indResult(statistic=1.2096277376026694, pvalue=0.22644804226361348)
# 4. **The p-value:** 0.226
# 5. **Conclusion:  P-value > Significance Level (0.226 > 0.05). We fail to reject Null Hypothesis**. Which means the average no. of cycles rented on workingday and on holiday is Euqal.

# In[184]:


#calculating mean and variance of workingday and holiday
workingday = df[df["workingday"]==1]
holiday = df[df["workingday"]==0]

print('Total no. of workingdays = '+ str(len(workingday)))
print('Total no. of holidays = '+ str(len(holiday)))
print()
print('Mean of workingdays:' + str(workingday["count"].mean()))
print('Mean of holiday:' + str(holiday["count"].mean()))
print()
print('Variance of workingdays:' + str(workingday["count"].var()))
print('Variance of holiday:' + str(holiday["count"].var()))


# In[244]:


#distplot to check the distribution of both workingday and holiday, and variance in the graph
f, ax = plt.subplots( figsize = (13,6) )  
sns.distplot(workingday['count'], color="r", ax = ax, label = "Workingday")
sns.distplot(holiday['count'], color="b", ax = ax, label = "Holiday")
plt.title("Cycles Rented distribution for Workingday and Holiday")
plt.legend()  
plt.show() 


# In[186]:


# significance level = 0.05
# test statistic: count on workingday and holiday
# caculating p-value and test statistics
st.ttest_ind(a=workingday["count"], b=holiday["count"], equal_var=True)


# ## `2. No. of cycles rented is similar or different in different Weather ?`

# ### `Insights and Proof:`
# 1. **Hypothesis formulation: H0: The average count of cycles rented in all 4 weathers is equal. H1: Atleast one weather's average cycle rented differs from the rest.**
# 2. **Test:**  ANOVA
# 3. **Result:** F_onewayResult(statistic=65.53024112793271, pvalue=5.482069475935669e-42)
# 4. **The p-value:** 5.482069475935669e-42
# 5. **Conclusion: P-value < Significance Level (5.482069475935669e-42 < 0.05). We Reject the Null Hypothesis. Which means one weather's average cycle rented differs from the rest.**

# In[187]:


#separating the data as per the weather
clear = df[df["weather"] == 1]
mist = df[df["weather"] == 2]
light_snow = df[df["weather"] == 3]
heavy_rain = df[df["weather"] == 4]

print('No. of Clear weather days = '+ str(len(clear)))
print('No. of Mist weather days = '+ str(len(mist)))
print('No. of Light Snow weather days = '+ str(len(light_snow)))
print('No. of Heavy Rain weather days = '+ str(len(heavy_rain)))
print()
print('Mean of Clear weather days:' + str(clear["count"].mean()))
print('Mean of Mist weather days:' + str(mist["count"].mean()))
print('Mean of Light Snow weather days:' + str(light_snow["count"].mean()))
print('Mean of Heavy Rain weather days:' + str(heavy_rain["count"].mean()))
print()
print('Variance of Clear weather days:' + str(clear["count"].var()))
print('Variance of Mist weather days:' + str(mist["count"].var()))
print('Variance of Light Snow weather days:' + str(light_snow["count"].var()))
print('Variance of Heavy Rain weather days:' + str(heavy_rain["count"].var()))


# In[243]:


#kdeplot to check the distribution of all Weathers
plt.figure(figsize=(13,6))
fig = sns.kdeplot(clear['count'], label = "Clear")
fig = sns.kdeplot(mist['count'], label = "Mist")
fig = sns.kdeplot(light_snow['count'], label = "Light Snow") 
fig = sns.kdeplot(heavy_rain['count'], label = "Heavy Rain")
plt.title("Cycles Rented distribution for 4 different Weathers")
plt.legend()  
plt.show()


# In[189]:


from scipy.stats import f_oneway
f_oneway(clear["count"], mist["count"], light_snow["count"], heavy_rain["count"])


# ## `3. No. of cycles rented is similar or different in different Seasons ?`

# ### `Insights and Proof:`
# 1. **Hypothesis formulation:** H0: Average Cyles rented in all seasons is equal. H1: Atleast one season's average rented cycles differs from the rest.
# 2. **Test : One-way ANOVA**
# 3. **Result:** F_onewayResult(statistic=236.94671081032106, pvalue=6.164843386499654e-149)
# 4. **The p-value:** 6.164843386499654e-149
# 5. **Conclusion: P-value < Significance Level (6.164843386499654e-149 < 0.05). We Reject the Null Hypothesis, which means one season's average cycle rented differs from the rest.**

# In[190]:


#separating the data as per the Seasons
spring = df[df["season"] == 1]
summer = df[df["season"] == 2]
fall = df[df["season"] == 3]
winter = df[df["season"] == 4]

print('No. of Spring days = '+ str(len(spring)))
print('No. of Summer days = '+ str(len(summer)))
print('No. of Fall days = '+ str(len(fall)))
print('No. of Winter days = '+ str(len(winter)))
print()
print('Mean of Spring days:' + str(spring["count"].mean()))
print('Mean of Summer days:' + str(summer["count"].mean()))
print('Mean of Fall days:' + str(fall["count"].mean()))
print('Mean of Winter days:' + str(winter["count"].mean()))
print()
print('Variance of Spring days:' + str(spring["count"].var()))
print('Variance of Summer days:' + str(summer["count"].var()))
print('Variance of Fall days:' + str(fall["count"].var()))
print('Variance of Winter days:' + str(winter["count"].var()))


# In[242]:


#distplot to check the distribution of all Seasons
f, ax = plt.subplots( figsize = (13,6) )  
sns.distplot(spring['count'], ax = ax, label = "Spring")
sns.distplot(summer['count'], ax = ax, label = "Summer")
sns.distplot(fall['count'], ax = ax, label = "Fall")
sns.distplot(winter['count'], ax = ax, label = "Winter")
plt.title("Cycles Rented distribution for 4 different Seasons")
plt.legend()  
plt.show()


# In[192]:


f_oneway(spring['count'], summer['count'], fall['count'], winter['count'])


# ## `4. Weather is dependent on the Season ?`

# ### `Insights and Proof:`
# 1. **Hypothesis formulation: H0: There is no relation between Weather and Season. H1: There is a significant relation between Weather and Season.**
# 2. **Test : Chi-Square Test**
# 3. **Result:** p-value is 1.5499250736864862e-07
# 4. **The p-value:** p-value is 1.5499250736864862e-07
# 5. **Conclusion: There is a significant relation between Weather and Season (Reject H0)**

# In[193]:


df1 = df[df['weather']!=4]
crosstab = pd.crosstab(df1["season"], df1["weather"])


# In[194]:


# calculating the p-value
from scipy.stats import chi2_contingency
crosstab = pd.crosstab(df["weather"],df["season"])
stat, p, dof, expected = chi2_contingency(crosstab)

alpha = 0.05
print("p-value is " + str(p))
if p <= alpha:
    print("There is a significant relation between Weather and Season (Reject H0)")
else:
    print("There is no relation between Weather and Season (Fail to Reject H0)")


# # <span style='background :yellow' > Recommendations: </span>

# ### **`Workingday vs Holiday`**
# 1. The cycle demand does not change even if it's a Holiday. Hence, keep similar or same offer on any day.
# 2. Inorder to get more demand on Holidays, some special offers can be **clubbed with tourist passes or tourist tickets.**
# 3. **A quarterly pass** which includes special offers on Holiday will help increase in demand.
# 
# ### **`Cycles Rented: Count`**
# 1. Average cylces rented is 191 and between 0 - 200, to increase it by some percent, registered users can be given few free rides after usage of a certain rides. **Eg: For registered users, on usage of 20 rides a month, get 3 rides free for the next month.**
# 
# ### **`Time:`**
# 1. Comapratively less cycles are rented between 8pm - 11pm, inorder to increase the usage at this hour **tie-ups with reataurants and delivery agents will help.**
# 2. **On holidays cycles can be made available at most touristy places of the city, at the places where people stroll around at night.**
# 
# ### **`Weather and Season`**:
# 1. As weather and season are related to each other, we can consider a particular season where less demand occurs. And bring **some exciting deals and improvement in rides.**
# 2. **Tie-up with Wildcraft** will bring some usage of cycles. Eg: People buying raincoats from Wildcraft can we offered come vouchers to be redeemed with Yulu cycles.
# 
# ### This sums up to final recommendation, **some innovation suggested would be offline map will help people to locate new/unaware places.**
# ### **An installed mobile-stand can be an add advantage.**
# 
