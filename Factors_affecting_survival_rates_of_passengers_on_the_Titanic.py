
# coding: utf-8

# # Titanic Data
# 
# ## What factors meant that people were more likely to survive? 
# 
# It is relatively well known that a higher percentage of women passengers survived, but are there other factors, such as class, or age, that were also important in the likelyhood that a passenger would survive?
# 
# Below is an exploration of the passenger information provided, from 891 of the 2224 passengers and crew onboard the Titanic.

# In[1]:

# Importing necessary libraries and reading in the csv file
import pandas as pd
import numpy as np

titanic_data = pd.read_csv('titanic-data.csv')

titanic_data.head()  # to get an idea of the data included in the file


# ### Male versus female survival rates
# 
# First I will explore the effect of gender on survival rates, using the .groupby function to group by gender.

# In[2]:

#using .count to find out how many males and female in the data set, and to get an idea about the completeness of the data.

titanic_grouped_by_sex = titanic_data.groupby('Sex')

titanic_grouped_by_sex.count()  


# Looking at the above table it seems that out of the 891 passengers in the dataset, 314 were female and 577 male. Most columns of the dataset are complete, except some ages are unknown, most of the cabins of the passengers are unknown, and there are two missing values for the port at which two female passengers embarked.
# 
# The 'Survived' column is made up of '1' or '0' depending upon whether the passenger survived or not. By adding up the values in this column, the number of passengers that survived can be found.

# In[3]:

titanic_m_f_survivors = titanic_data.groupby('Sex')['Survived'].sum()

titanic_m_f_survivors


# In[4]:

percentage_m_f_survivors = (titanic_m_f_survivors.div(titanic_grouped_by_sex['PassengerId'].count()))*100

percentage_m_f_survivors


# Out of the dataset provided, there were a total of 342 survivors, giving a survival rate of just over 38 %. Of these, 233 were female passengers, accounting for approximately 74 % of the female passengers onboard the ship, whilst only 109 were male passengers, accounting for just under 19 % of the male passengers onboard. 
# 
# Put another way, although male passengers made up nearly 65 % of the total passengers onboard the Titanic, just under 32 % of the survivors were male.

# ### Did ticket class affect a passenger's survival rate?
# 
# I was interested to know more about this data set, specifically, within the male and female survivor groups, how did the demographic vary in terms of ticket class.

# In[5]:

# First I looked at the class breakdown of all passengers onboard, grouped by male and female
titanic_classes_numbers = titanic_data.groupby(['Sex', 'Pclass'])['PassengerId'].count()
print (titanic_classes_numbers)

titantic_classes_percent = (titanic_classes_numbers.div(titanic_grouped_by_sex['PassengerId'].count()))*100
titantic_classes_percent


# In[6]:

# I then looked at the class breakdown of the survivors, grouped by male and female
titanic_classes_numbers_survived = titanic_data.groupby(['Sex', 'Pclass'])['Survived'].sum()
print (titanic_classes_numbers_survived)

titantic_classes_percent_survived = (titanic_classes_numbers_survived.div(titanic_m_f_survivors))*100
titantic_classes_percent_survived


# In both the male and females groups, a higher proportion of first class passengers were saved, compared to the proportion of first class passengers onboard. Interestingly, 45 out of the 109 male survivors were travelling in first class, 41 % of the male survivors, even though only 21 % of male passengers were travelling in this class.
# 
# The differences between total number of passengers and the number of survivors from each class can be more easily communicated using a visualisation.

# In[7]:

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[8]:

#function to define style of pie chart subplots
def plot_subplot(data):
    return plt.pie(data, colors=['navy', 'powderblue', 'darkturquoise'], labels=['first', 'second', 'third'], labeldistance=1.1, startangle=90, counterclock=False)


# In[9]:

plt.figure(figsize=(8, 8))
plt.subplot(221, title='Ticket classes of the \n female passengers on board', aspect=1)
plot_subplot(titantic_classes_percent['female'])

plt.subplot(222, title='Ticket classes of the \n male passengers on board', aspect=1)
plot_subplot(titantic_classes_percent['male'])

plt.subplot(223, title='Ticket classes of the \n female passengers that survived', aspect=1)
plot_subplot(titantic_classes_percent_survived['female'])

plt.subplot(224, title='Ticket classes of the \n male passengers that survived', aspect=1)
plot_subplot(titantic_classes_percent_survived['male'])


# By comparing the visualisations above, it can be seen that although the largest proprotion of passengers in both the male and female groups were travelling in third class, this proportion shrinks when compared to the ticket class of the survivors. Those that travelled in first class made up a far higher percentage of the survivors then would be reflected by the proportion of first class passengers.

# ### Did age affect a passenger's survival rate?
# Continuing to explore the data, I studied whether the age of a passenger reflected how likely the passenger was to survive.

# In[10]:

#using the .describe function to get a general idea about the data
titanic_ages = titanic_data.groupby(['Sex', 'Survived'])['Age']

titanic_ages.describe()


# Looking at the data shown above, it seems that the mean and quartile ranges do not vary significantly between those that survived and those that did not. The standard deviation of each dataset is relatively large at between 13.6 and 16.5. 
# 
# To futher check for any relationships between survival and age, I plotted the data in the form of a number of histograms to help visualise the age distribution of the passengers.

# In[11]:

#creating histograms to look at the distribution of ages amongst all on board the Titanic and also amongst
#those that survived and those that did not.
#Data is first cleaned using .dropna() to remove any NaN values.

titanic_data['Age'].dropna().hist(bins=16, range=(0,80));
plt.title('Age demographic of passengers onboard the Titanic')
titanic_data['Age'].dropna().hist(by=titanic_data['Survived'], bins=16, range=(0,80))
plt.suptitle('Age demographic amongst non-survivors (graph 0) \n and survivors (graph 1) of the Titanic disaster', y=1.05)


# Comparing the graphs above it can be seen that for those aged under 10, more passengers survived than did not, something which is not true for any other age group. Other than this, both the graphs for those that survived and those that did not show a similar distribution of ages to the general population of passengers aboard the ship. 
# 
# To investigate further the survival rates of the children onboard the titanic, I was interested in looking further into the data to explore if those with parents or siblings on board (identified in the dataset by 'Parch' > 0) were more likely to survive.
# 
# ### Did the number of siblings and parents a passenger had onboard affect the likelyhood of survival?
# 
# Below I have grouped the data by 'Parch' to explore this question.

# In[12]:

#first looking at the number of entries in the dataset with Parch > 0
titanic_grouped_by_parch = titanic_data.groupby('Parch')['PassengerId'].count()

titanic_grouped_by_parch


# There were a total of 213 passengers onboard that also had at least one parent and/or sibling onboard.

# In[13]:

titanic_parch_and_survived = titanic_data.groupby('Parch')['Survived'].sum()

titanic_parch_and_survived 


# In[14]:

survival_rates_by_parch = (titanic_parch_and_survived / titanic_grouped_by_parch)*100
survival_rates_by_parch


# Within this dataset, the overall survival rate for the passengers onboard the Titanic was just over 38 %. Comparing this value to the data above, it does seem as if having 1, 2 or 3 parents and/or siblings also onboard did increase ones chances of survival. However, having more than this resulted in the opposite effect.

# ## Conclusions
# 
# This report has explored how the gender, ticket class and age of a passenger aboard the Titanic, as well as the number of parents and/or children also onboard, reflected the likelihood that a passenger would have survived the disaster. 
# 
# No statistical analysis has been undertaken, and so no conclusions can be made as to the statistic significance of the findings, however, interesting conclusions can be made nonetheless.
# 
# Female passengers were far more likely than their male counterparts to survive, with 74 % of the female passengers onboard the ship surviving, whilst just under 19 % of the male passengers did so. Amongst both the male and female passengers, those traveling in first class were more likely to survive than those travelling in third.
# 
# The likelihood of survival was less correlated to age, apart from those who were aged less than 10 for who there did seem to be some advantage. This led me to explore if there was a relationship between travelling with parents and/or siblings and survival. It seems that travelling with between 1 and 3 close relatives did indeed increase the likelyhood of survival, however, for numbers above this the opposite seemed to be true.

# ## Note:
# 
# No other resourses were used in creating this submission.
