#!/usr/bin/env python
# coding: utf-8

# # COMP4605.1 Data Mining Project
# In this project I will be using e-commerce rates per years in some countries and determine if there is a raise in the ecommerce rate after the pandemic happened. 
# My gathered data is between the years 2013-2021. I will make the analysis of the data respectively and compare the countries within each other to find out with country has the biggest rate of going up in ecommerce.
# First lets import the librarys we will use.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
countries = pd.read_csv('countries.csv',error_bad_lines=False,encoding = "utf-8")
categories = pd.read_csv('categoriess.csv',error_bad_lines=False,encoding = "utf-8")
uk = pd.read_csv('UK.csv',error_bad_lines=False,encoding = "utf-8")
countries


# In[2]:


countries.info()


# There are 19 countries, and 9 years that we will observe the ecommerce rate differing in between. This is the shape of our dataset

# In[3]:


countries.describe()


# In[4]:


countriesdata = countries.fillna("Unknown")
#we will fill the NaN's with "Unknown" as those years' ecommerce rates are unknown. 
countriesdata


# In[5]:


co = countriesdata['Country']
years = np.arange(2013, 2022).tolist()
print("Years:",years)


# In[6]:


data = np.array([co, years], dtype="object")
data = data.T
print(data[0:5])


# In[7]:


countries.hist(figsize=(10,8),bins=20)

years = np.arange(2013, 2021).tolist()

countries = pd.read_csv('countries.csv', index_col='Country') #set column as index
countries = countries.T #transpose so now our rows are our columns
countries.plot(legend=False, marker='o')
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=19)
plt.show()
# We see the e-commerce rate in each country according to years in this plot

# In[9]:


#lets check the correlation between years
countries = countries.T
countriescorr =countries.corr()
countriescorr


# In[10]:


plt.figure(figsize = (10,7))
sns.heatmap(countriescorr, annot = True)
plt.show()


# In[11]:


pd.plotting.scatter_matrix(countriescorr, figsize=(25,25), s=300, alpha=0.7)
plt.show()


# So we see a nearly perfect correlation between 2018-2019. Which is when the pandemic started. So we see that the pandemic's start had effected the ecommerce rates. Now let's take a look countries one by one and examine which country has the greatest rate going up.

# In[12]:


countries = countries.T
turkey = countries.Turkey.plot(legend=True, marker='o', markerfacecolor='white', color='red') #plot turkey column
turkey
mint = min(countries["Turkey"])
maxt = max(countries["Turkey"])
meant = (countries["Turkey"]).mean()
print("The range between the max percentage and the lowest in Turkey: ", maxt-mint )
print("The mean of the ecommerce rates of Turkey is: ", meant)
print(countries["Turkey"].describe())


# In[13]:


countries.UK.plot(legend=True, marker='o',  markerfacecolor='red' )
minuk = min(countries["UK"])
maxuk = max(countries["UK"])
meanuk = (countries["UK"]).mean()
print("The range between the max percentage and the lowest in UK: ", maxuk-minuk )
print("The mean of the ecommerce rates of UK is: ", meanuk)
print(countries["UK"].describe())


# In[14]:


countries.USA.plot(legend=True, marker="o",  markerfacecolor='blue',color='red')
minus = min(countries["USA"])
maxus = max(countries["USA"])
meanus = (countries["USA"]).mean()
print("The range between the max percentage and the lowest in USA: ", maxus-minus )
print("The mean of the ecommerce rates of USA is: ",meanus)
print(countries["USA"].describe())


# In[15]:


countries.Germany.plot(legend=True, marker="o", color="black",  markerfacecolor='yellow')
ming = min(countries["Germany"])
maxg = max(countries["Germany"])
meang = (countries["Germany"]).mean()
print("The range between the max percentage and the lowest in Germany: ", maxg-ming )
print("The mean of the ecommerce rates of Germany is: ",meang)
print(countries["Germany"].describe())


# In[16]:


countries.France.plot(legend=True, marker="o",  markerfacecolor='white')
minf = min(countries["France"])
maxf = max(countries["France"])
meanf = (countries["France"]).mean()
print("The range between the max percentage and the lowest in France: ", maxf-minf )
print("The mean of the ecommerce rates of France is: ",meanf)
print(countries["France"].describe())


# In[17]:


countries.Japan.plot(legend=True, marker="o",  markerfacecolor='white', color="red")
minj = min(countries["Japan"])
maxj = max(countries["Japan"])
meanj = (countries["Japan"]).mean()
print("The range between the max percentage and the lowest in Japan: ", maxj-minj )
print("The mean of the ecommerce rates of Japan is: ",meanj)
print(countries["Japan"].describe())


# In[18]:


countries.Italy.plot(legend=True, marker="o", color="green",  markerfacecolor='red')
mini = min(countries["Italy"])
maxi = max(countries["Italy"])
meani = (countries["Italy"]).mean()
print("The range between the max percentage and the lowest in Italy: ", maxi-mini )
print("The mean of the ecommerce rates of Italy is: ",meani)
print(countries["Italy"].describe())


# In[19]:


countries.Spain.plot(legend=True, marker="o", color="gold",  markerfacecolor='red')
mins = min(countries["Spain"])
maxs = max(countries["Spain"])
means = (countries["Spain"]).mean()
print("The range between the max percentage and the lowest in Spain: ", maxs-mins )
print("The mean of the ecommerce rates of Spain is: ",means)
print(countries["Spain"].describe())


# In[20]:


countries.China.plot(legend=True, marker="o", color="red",  markerfacecolor='yellow')
minc = min(countries["China"])
maxc = max(countries["China"])
meanc = (countries["China"]).mean()
print("The range between the max percentage and the lowest in China: ", maxc-minc )
print("The mean of the ecommerce rates of China is: ",meanc)
print(countries["China"].describe())


# In[21]:


countries.Poland.plot(legend=True, marker="o", color="red",  markerfacecolor='white')
minp= min(countries["Poland"])
maxp= max(countries["Poland"])
meanp = (countries["Poland"]).mean()
print("The range between the max percentage and the lowest in Poland: ", maxp-minp )
print("The mean of the ecommerce rates of Poland is: ",meanp)
print(countries["Poland"].describe())


# In[22]:


countries.India.plot(legend=True, marker="o", color="green",  markerfacecolor='black')
minin= min(countries["India"])
maxin= max(countries["India"])
meanin = (countries["India"]).mean()
print("The range between the max percentage and the lowest in India: ", maxin-minin )
print("The mean of the ecommerce rates of India is: ",meanin)
print(countries["India"].describe())


# In[23]:


countries.Brazil.plot(legend=True, marker="o", color="green",  markerfacecolor='yellow')
minb= min(countries["Brazil"])
maxb= max(countries["Brazil"])
meanb = (countries["Brazil"]).mean()
print("The range between the max percentage and the lowest in Brazil: ", maxb-minb )
print("The mean of the ecommerce rates of Brazil is: ",meanb)
print(countries["Brazil"].describe())


# In[24]:


countries.Russia.plot(legend=True, marker="o")
minr= min(countries["Russia"])
maxr= max(countries["Russia"])
meanr = (countries["Russia"]).mean()
print("The range between the max percentage and the lowest in Russia: ", maxr-minr )
print("The mean of the ecommerce rates of Russia is: ",meanr)
print(countries["Russia"].describe())


# In[25]:


countries.Netherlands.plot(legend=True, marker="o",  markerfacecolor='darkred')
minne= min(countries["Netherlands"])
maxne= max(countries["Netherlands"])
meanne = (countries["Netherlands"]).mean()
print("The range between the max percentage and the lowest in Netherlands: ", maxne-minne )
print("The mean of the ecommerce rates of Netherlands is: ",meanne)
print(countries["Netherlands"].describe())


# In[26]:


countries.Nigeria.plot(legend=True, marker="o", color="green",  markerfacecolor='white')
minn = min(countries["Nigeria"])
maxn = max(countries["Nigeria"])
meann = (countries["Nigeria"]).mean()
print("The range between the max percentage and the lowest in Nigeria: ", maxn-minn )
print("The mean of the ecommerce rates of Nigeria is: ",meann)
print(countries["Nigeria"].describe())


# In[27]:


countries.Belgium.plot(legend=True,marker="o", color="black",  markerfacecolor='yellow')
minb = min(countries["Belgium"])
maxb = max(countries["Belgium"])
meanb = (countries["Belgium"]).mean()
print("The range between the max percentage and the lowest in Belgium: ", maxb-minb )
print("The mean of the ecommerce rates of Belgium is: ",meanb)
print(countries["Belgium"].describe())


# In[28]:


countries.Luxembourg.plot(legend=True, marker="o", color="red",  markerfacecolor='blue')
minl = min(countries["Luxembourg"])
maxl = max(countries["Luxembourg"])
meanl = (countries["Luxembourg"]).mean()
print("The range between the max percentage and the lowest in Luxembourg: ", maxl-minl )
print("The mean of the ecommerce rates of Luxembourg is: ",meanl)
print(countries["Luxembourg"].describe())


# In[29]:


countries.Korea.plot(legend=True, marker="o",  markerfacecolor='black')
mink = min(countries["Korea"])
maxk = max(countries["Korea"])
meank = (countries["Korea"]).mean()
print("The range between the max percentage and the lowest in Korea: ", maxk-mink )
print("The mean of the ecommerce rates of Korea is: ",meank)
print(countries["Korea"].describe())


# In[30]:


countries.Cyprus.plot(legend=True, marker="o", color="orange",  markerfacecolor='black')
mincy = min(countries["Cyprus"])
maxcy = max(countries["Cyprus"])
meancy = (countries["Cyprus"]).mean()
print("The range between the max percentage and the lowest in Cyprus: ", maxcy-mincy )
print("The mean of the ecommerce rates of Cyprus is: ",meancy)
print(countries["Cyprus"].describe())


# From the mean and the range we see that China is in the lead. Let's take a look at the Standard Errors.

# In[31]:


print("SEM for Turkey", stats.sem(countries.Turkey))
print("SEM for UK", stats.sem(countries.UK))
print("SEM for USA", stats.sem(countries.USA))
print("SEM for Germany", stats.sem(countries.Germany))
print("SEM for France", stats.sem(countries.France))
print("SEM for Japan", stats.sem(countries.Japan))
print("SEM for Italy", stats.sem(countries.Italy))
print("SEM for Spain", stats.sem(countries.Spain))
print("SEM for China", stats.sem(countries.China))
print("SEM for Poland", stats.sem(countries.Poland))
print("SEM for India", stats.sem(countries.India))
print("SEM for Brazil", stats.sem(countries.Brazil))
print("SEM for Russia", stats.sem(countries.Russia))
print("SEM for Netherlands", stats.sem(countries.Netherlands))
print("SEM for Nigeria", stats.sem(countries.Nigeria))
print("SEM for Belgium", stats.sem(countries.Belgium))
print("SEM for Luxembourg", stats.sem(countries.Luxembourg))
print("SEM for Korea", stats.sem(countries.Korea))
print("SEM for Cyprus", stats.sem(countries.Cyprus))


# # Hypothesis 1
# “Ecommerce rates over countries. Which country has the greatest rate going up?”
# Hypothesis: U.S. has the greatest rate going up in pandemic.
# 
# SO, let's take a look at USA too.

# In[36]:


plt.hist(countries.China)


# In[160]:


plt.hist(countries.USA)


# China has the highest numbers through the years but we are looking for the highest going up in percentages. Let's take a look at UK.

# In[41]:


plt.hist(countries.UK)


# In[42]:


countries.China.plot(legend=True, marker="o", color="red",  markerfacecolor='yellow')
countries.USA.plot(legend=True, marker="o",  markerfacecolor='blue',color='red')
countries.UK.plot(legend=True, marker='o',  markerfacecolor='red' )


# From the countries examined one by one we know the max ecommerce rates which are as followed:

# In[137]:


print("Max for China: ", maxc)
print("Max for UK: ", maxuk)
print("Max for USA: ", maxus)
#we will compare their max rates with the year before.
print("Max value for China is in year: ",countries.China.idxmax())
print("The year before has the percentage: ", countries.China.iloc[5:6])
print("The rate of going up in China is 4.5")
print("Max value for USA is in year: ",countries.USA.idxmax())
print("The year before has the percentage: ", countries.USA.iloc[6:7])
print("The rate of going up in USA is 6.1")
print("Max value for UK is in year: ",countries.UK.idxmax())
print("The year before has the percentage: ", countries.UK.iloc[6:7])
print("The rate of going up in UK is 11")


# In[175]:


britain = countries.UK
usa = countries.USA
data = np.array([britain, usa])
data = data.T
plt.boxplot(data)
plt.grid()
plt.plot(1,6,'ro')


# In[159]:


#Let's check this with t-test.

britain = countries.UK
usa = countries.USA
from scipy.stats import ttest_ind, ttest_ind_from_stats
degrees_of_freedom =  len(countries['UK'] + len(countries['USA']) ) - 2
print('degrees of freedom=',degrees_of_freedom)
alpha = 0.05
two_tailed_test_prob_tail = alpha/2
t_critical = round(stats.t.ppf(two_tailed_test_prob_tail, degrees_of_freedom), 3)
print('t critical',t_critical)
stat, p = ttest_ind(countries['UK'], countries['USA'])
print('t=%.3f, p=%.3f' % (stat, p))

if abs(stat) <= t_critical:
     print('Accept null hypothesis.')
else:
     print('Reject the null hypothesis.')


# H0: USA has the greatest rate going up.
# 
# We come to the conclusion that we reject H0, as UK has a biggest rate.

# # Hypothesis 2
# 
#  “Did covid effect ecommerce in a good, or bad way?” is the main question in my project. 
#  Hypothesis: It effected ecommerce in a good way
# 
# From the plots we have examined, we are confident to say that pandemic has effected ecommerce in a good way

# # Hypothesis 3
# 
# “What will be the e-commerce grow rate in 2020 in Turkey according to the given data? 
# Will the e-commerce rate go up or down?”
# 
# Hypothesis: The rate will go up 
# 
# So, let's take a look at Turkey, again.

# In[44]:


countries.Turkey.plot(legend=True, marker="o")


# In[45]:


from sklearn.linear_model import LinearRegression
years = np.arange(2013, 2022).tolist()
lr = LinearRegression()
turkey = countries["Turkey"].values.reshape(-1,1)
lr.fit(turkey, years)
pred = lr.predict(turkey)
plt.plot(turkey, years, 'rs')
plt.plot(turkey, pred, 'b--')
plt.xlabel('Turkey E-commerce Rate')
plt.ylabel('Years')
plt.grid()


# In[46]:


#Now we will use the last 3 years 2019-2020-2021 to predict the next 3 years
recent3 = turkey[-3:].reshape(-1,1)
prednext3 = lr.predict(recent3)
plt.plot(turkey, years, 'rs')
plt.plot(turkey, pred, 'b--')
plt.xlabel('Turkey E-commerce Rate')
plt.ylabel('Years')
plt.plot(recent3, prednext3, 'g-X')
plt.grid()


# So, we won't reject H0 here as the rates went up. Let's take a look at our next data. 
# # The categories

# In[47]:


categories


# In[48]:


categories.describe()


# In[49]:


categories.info()


# In[50]:


from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

categories = categories.fillna("0")
categories = categories.drop(columns=['Categories'])


# In[51]:


km = KMeans(n_clusters=15, init='k-means++',random_state=0)
km.fit(categories)
predict = km.predict(categories)
#print(km.cluster_centers_)


# In[52]:


centers = km.cluster_centers_
print(centers)


# In[53]:


print(predict)


# In[54]:


cat = categories.values
plt.scatter(cat[:, 0], cat[:, 1], c=predict, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# Now let's take a look at the next dataset where we will comment on the raise or drop in the categories. The next dataset is from UK. 
# # Hypothesis 4 
#  “Did ecommerce rates go up in each 
# category?”
# Hypothesis: Ecommerce rates went up in each category
# 

# In[55]:


uk


# In[56]:


uk.describe()


# In[57]:


uk.info()


# In[58]:


uk["Non-store retailing"].hist( color = "pink")
plt.title('Non-store retailing')
plt.show()
uk["Textiles clothing and footwear stores"].hist(color = "red")
plt.title("Textiles clothing and footwear stores")
plt.show()
uk["All retailing"].hist()
plt.title("All retailing")
plt.show()
uk["Household goods stores"].hist(color = "green")
plt.title("Household goods stores ")
plt.show()
uk["Predominantly food stores"].hist(color = "purple")
plt.title("Predominantly food stores")
plt.show()
uk["Other stores"].hist(color = "orange")
plt.title("Other stores")
plt.show()


# Let's take a look at the means

# In[59]:


m1 = uk["Non-store retailing"].mean()
print("Mean for Non-store retailing is: ", m1)
m2 = uk["Textiles clothing and footwear stores"].mean()
print("Mean for Textiles clothing and footwear stores is: ",m2)
m3 = uk["All retailing"].mean()
print("Mean for All retailing is: ", m3)
m4 = uk["Household goods stores"].mean()
print("Mean for Household goods stores is: ", m4)
m5 = uk["Predominantly food stores"].mean()
print("Mean for Predominantly food stores is: ",m5)
m6 = uk["Other stores"].mean()
print("Mean for Other stores is: ",m6)


# In[60]:


ukdate = uk["Date"]
uk = uk.drop(columns=['Date'])
uk = uk.drop(columns=['Data Type'])


# In[61]:


km = KMeans(n_clusters=6, init='k-means++',random_state=0)
km.fit(uk)
predict = km.predict(uk)
centers = km.cluster_centers_
print(centers)


# In[62]:


print(predict)


# In[63]:


val = uk.values
plt.scatter(val[:, 0], val[:, 1], c=predict, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[64]:


uk.plot(legend=False)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=6)
plt.show()


# In[65]:


plt.scatter(uk["Non-store retailing"],ukdate)
plt.title("Non-store retailing")
plt.show()
plt.scatter(uk["Textiles clothing and footwear stores"], ukdate)
plt.title("Textiles clothing and footwear stores")
plt.show()
plt.scatter(uk["All retailing"], ukdate)
plt.title("All retailing")
plt.show()
plt.scatter(uk["Household goods stores"], ukdate)
plt.title("Household goods stores ")
plt.show()
plt.scatter(uk["Predominantly food stores"], ukdate)
plt.title("Predominantly food stores")
plt.show()
plt.scatter(uk["Other stores"], ukdate)
plt.title("Other stores")
plt.show()


# In[101]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
ukdate = np.arange(1, 136).tolist() #136 as there are 136 rows.
ukdate = np.array(ukdate).reshape(-1,1)
X_non =  np.array(uk["Non-store retailing"]).reshape(-1,1)
lin_reg_non = LinearRegression()
lin_reg_non.fit(ukdate,X_non)
#Polynomial Regression Model Fitting
poly_reg_non = PolynomialFeatures(degree=3)
X_poly_non = poly_reg_non.fit_transform(ukdate)
pol_reg_non = LinearRegression()
pol_reg_non.fit(X_poly_non, X_non)
linear_pred_non = lin_reg_non.predict(ukdate)
poly_pred_non = pol_reg_non.predict(poly_reg_non.fit_transform(ukdate))

plt.scatter(ukdate, X_non)
plt.plot(ukdate, linear_pred_non, '+-', color='black')
plt.plot(ukdate, poly_pred_non, '+-', color='red')
plt.title('Non-store Retailing Rate')
plt.xlabel('Date')
plt.ylabel('Non-store Retailing')
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[103]:


print('Score:', mean_squared_error(X_non, linear_pred_non))
print('Score:', mean_squared_error(X_non, poly_pred_non))
print('Linear Model Fit Score:', lin_reg_non.score(ukdate, X_non))
print('Polynomial Model Fit Score:', pol_reg_non.score(X_poly_non, X_non))


#  We can say that polynomial regression is more dependable than linear regression

# In[104]:


print('Linear Regression Non-Store Retailing Predictions')
print('20 Nov: ', lin_reg_non.predict([[131]]))
print('20 Dec: ', lin_reg_non.predict([[132]]))
print('21 Jan: ', lin_reg_non.predict([[133]]))
print('21 Feb: ', lin_reg_non.predict([[134]]))
print('21 Mar: ', lin_reg_non.predict([[135]]))


# In[105]:


print('Polynomial Regression  Non-Store Retailing Predictions')
print('20 Nov: ', pol_reg_non.predict(poly_reg_non.fit_transform([[131]])))
print('20 Dec: ', pol_reg_non.predict(poly_reg_non.fit_transform([[132]])))
print('21 Jan: ', pol_reg_non.predict(poly_reg_non.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_non.predict(poly_reg_non.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_non.predict(poly_reg_non.fit_transform([[135]])))


# ### Real Results
# 20-Oct	83.8
# 
# 20-Nov	82.6
# 
# 20-Dec	79.3
# 
# 21-Jan	83
# 
# 21-Feb	85
# 
# 21-Mar	83.2
# 
# The predicted values aren't so close to the real values, but they also can't be considered far either. Our error rate for the prediction was close to zero but that number for the test predictions. The predictions can be trusted to a certain value

# In[111]:


X_text =  np.array(uk["Textiles clothing and footwear stores"]).reshape(-1,1)
lin_reg_text = LinearRegression()
lin_reg_text.fit(ukdate,X_text)
#Polynomial Regression Model Fitting
poly_reg_text = PolynomialFeatures(degree=3)
X_poly_text = poly_reg_text.fit_transform(ukdate)
pol_reg_text = LinearRegression()
pol_reg_text.fit(X_poly_text, X_text)
linear_pred_text = lin_reg_text.predict(ukdate)
poly_pred_text = pol_reg_text.predict(poly_reg_text.fit_transform(ukdate))

plt.scatter(ukdate, X_text)
plt.plot(ukdate, linear_pred_text, '+-', color='black')
plt.plot(ukdate, poly_pred_text, '+-', color='red')
plt.title("Textiles clothing and footwear stores Rate")
plt.xlabel('Date')
plt.ylabel("Textiles clothing and footwear stores")
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[112]:


print('Score:', mean_squared_error(ukdate, linear_pred_text))
print('Score:', mean_squared_error(ukdate, poly_pred_text))
print('Linear Model Fit Score:', lin_reg_text.score(X_text, ukdate))
print('Polynomial Model Fit Score:', pol_reg_text.score(X_poly_text, ukdate))


# In[113]:


print('Linear Regression Textiles clothing and footwear stores Predictions')
print('20 Nov: ', lin_reg_text.predict([[131]]))
print('20 Dec: ', lin_reg_text.predict([[132]]))
print('21 Jan: ', lin_reg_text.predict([[133]]))
print('21 Feb: ', lin_reg_text.predict([[134]]))
print('21 Mar: ', lin_reg_text.predict([[135]]))


# In[122]:


print('Polynomial Regression Textiles clothing and footwear stores Predictions')
print('20 Nov: ', pol_reg_text.predict(poly_reg_text.fit_transform([[131]])))
print('20 Dec: ', pol_reg_text.predict(poly_reg_text.fit_transform([[132]])))
print('21 Jan: ', pol_reg_text.predict(poly_reg_text.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_text.predict(poly_reg_text.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_text.predict(poly_reg_text.fit_transform([[135]])))


# ### Real Results
# 20-Oct	29.7
# 
# 20-Nov	41.8
# 
# 20-Dec	32.4
# 
# 21-Jan	50.8
# 
# 21-Feb	59.5
# 
# 21-Mar	55.7
# 

# In[115]:


X_all =  np.array(uk["All retailing"]).reshape(-1,1)
lin_reg_all = LinearRegression()
lin_reg_all.fit(ukdate,X_all)
#Polynomial Regression Model Fitting
poly_reg_all = PolynomialFeatures(degree=3)
X_poly_all = poly_reg_all.fit_transform(ukdate)
pol_reg_all = LinearRegression()
pol_reg_all.fit(X_poly_all, X_all)
linear_pred_all = lin_reg_all.predict(ukdate)
poly_pred_all = pol_reg_all.predict(poly_reg_all.fit_transform(ukdate))

plt.scatter(ukdate, X_all)
plt.plot(ukdate, linear_pred_all, '+-', color='black')
plt.plot(ukdate, poly_pred_all, '+-', color='red')
plt.title("All retailing")
plt.xlabel('Date')
plt.ylabel("All retailing")
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[118]:


print('Score:', mean_squared_error(X_all, linear_pred_all))
print('Score:', mean_squared_error(X_all, poly_pred_all))
print('Linear Model Fit Score:', lin_reg_all.score(ukdate, X_all))
print('Polynomial Model Fit Score:', pol_reg_all.score(X_poly_all, X_all))


# In[119]:


print('Linear Regression All Retailing Predictions')
print('20 Nov: ', lin_reg_all.predict([[131]]))
print('20 Dec: ', lin_reg_all.predict([[132]]))
print('21 Jan: ', lin_reg_all.predict([[133]]))
print('21 Feb: ', lin_reg_all.predict([[134]]))
print('21 Mar: ', lin_reg_all.predict([[135]]))


# In[121]:


print('Polynomial Regression  All Retailing Predictions')
print('20 Nov: ', pol_reg_all.predict(poly_reg_all.fit_transform([[131]])))
print('20 Dec: ', pol_reg_all.predict(poly_reg_all.fit_transform([[132]])))
print('21 Jan: ', pol_reg_all.predict(poly_reg_all.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_all.predict(poly_reg_all.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_all.predict(poly_reg_all.fit_transform([[135]])))


# ### Real Results
# 20-Oct	29.2
# 
# 20-Nov	32
# 
# 20-Dec	29.8
# 
# 21-Jan	35.3
# 
# 21-Feb	36.2
# 
# 21-Mar	34.7

# In[123]:


X_house =  np.array(uk["Household goods stores"]).reshape(-1,1)
lin_reg_house = LinearRegression()
lin_reg_house.fit(ukdate,X_house)
#Polynomial Regression Model Fitting
poly_reg_house = PolynomialFeatures(degree=3)
X_poly_house = poly_reg_house.fit_transform(ukdate)
pol_reg_house = LinearRegression()
pol_reg_house.fit(X_poly_house, X_house)
linear_pred_house= lin_reg_house.predict(ukdate)
poly_pred_house = pol_reg_house.predict(poly_reg_house.fit_transform(ukdate))

plt.scatter(ukdate, X_house)
plt.plot(ukdate, linear_pred_house, '+-', color='black')
plt.plot(ukdate, poly_pred_house, '+-', color='red')
plt.title("Household goods stores")
plt.xlabel('Date')
plt.ylabel("Household goods stores")
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[124]:


print('Score:', mean_squared_error(X_house, linear_pred_all))
print('Score:', mean_squared_error(X_house, poly_pred_all))
print('Linear Model Fit Score:', lin_reg_all.score(ukdate, X_house))
print('Polynomial Model Fit Score:', pol_reg_all.score(X_poly_house, X_house))


# In[125]:


print('Linear Regression Household goods stores Predictions')
print('20 Nov: ', lin_reg_house.predict([[131]]))
print('20 Dec: ', lin_reg_house.predict([[132]]))
print('21 Jan: ', lin_reg_house.predict([[133]]))
print('21 Feb: ', lin_reg_house.predict([[134]]))
print('21 Mar: ', lin_reg_house.predict([[135]]))


# In[126]:


print('Polynomial Regression   Household goods stores Predictions')
print('20 Nov: ', pol_reg_house.predict(poly_reg_house.fit_transform([[131]])))
print('20 Dec: ', pol_reg_house.predict(poly_reg_house.fit_transform([[132]])))
print('21 Jan: ', pol_reg_house.predict(poly_reg_house.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_house.predict(poly_reg_house.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_house.predict(poly_reg_house.fit_transform([[135]])))


# ### Real Results
# 20-Oct	23.9
# 
# 20-Nov	28.6
# 
# 20-Dec	21.4
# 
# 21-Jan	32.6
# 
# 21-Feb	38.5
# 
# 21-Mar	36.7

# In[127]:


X_pre =  np.array(uk["Predominantly food stores"]).reshape(-1,1)
lin_reg_pre = LinearRegression()
lin_reg_pre.fit(ukdate,X_pre)
#Polynomial Regression Model Fitting
poly_reg_pre= PolynomialFeatures(degree=3)
X_poly_pre = poly_reg_pre.fit_transform(ukdate)
pol_reg_pre = LinearRegression()
pol_reg_pre.fit(X_poly_pre, X_pre)
linear_pred_pre= lin_reg_pre.predict(ukdate)
poly_pred_pre= pol_reg_pre.predict(poly_reg_pre.fit_transform(ukdate))

plt.scatter(ukdate, X_pre)
plt.plot(ukdate, linear_pred_pre, '+-', color='black')
plt.plot(ukdate, poly_pred_pre, '+-', color='red')
plt.title("Predominantly food stores")
plt.xlabel('Date')
plt.ylabel("Predominantly food stores")
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[130]:


print('Score:', mean_squared_error(X_pre, linear_pred_pre))
print('Score:', mean_squared_error(X_pre, poly_pred_pre))
print('Linear Model Fit Score:', lin_reg_pre.score(ukdate, X_pre))
print('Polynomial Model Fit Score:', pol_reg_pre.score(X_poly_pre, X_pre))


# In[131]:


print('Linear Regression Predominantly food stores Predictions')
print('20 Nov: ', lin_reg_pre.predict([[131]]))
print('20 Dec: ', lin_reg_pre.predict([[132]]))
print('21 Jan: ', lin_reg_pre.predict([[133]]))
print('21 Feb: ', lin_reg_pre.predict([[134]]))
print('21 Mar: ', lin_reg_pre.predict([[135]]))


# In[132]:


print('Polynomial Regression  Predominantly food stores Predictions')
print('20 Nov: ', pol_reg_pre.predict(poly_reg_pre.fit_transform([[131]])))
print('20 Dec: ', pol_reg_pre.predict(poly_reg_pre.fit_transform([[132]])))
print('21 Jan: ', pol_reg_pre.predict(poly_reg_pre.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_pre.predict(poly_reg_pre.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_pre.predict(poly_reg_pre.fit_transform([[135]])))


# ### Real Results
# 20-Oct	10.5
# 
# 20-Nov	10.5
# 
# 20-Dec	11.2
# 
# 21-Jan	12.1
# 
# 21-Feb	11.9
# 
# 21-Mar	11.7

# In[133]:


X_other =  np.array(uk["Other stores"]).reshape(-1,1)
lin_reg_other = LinearRegression()
lin_reg_other.fit(ukdate,X_other)
#Polynomial Regression Model Fitting
poly_reg_other = PolynomialFeatures(degree=3)
X_poly_other= poly_reg_other.fit_transform(ukdate)
pol_reg_other = LinearRegression()
pol_reg_other.fit(X_poly_other, X_other)
linear_pred_other = lin_reg_other.predict(ukdate)
poly_pred_other = pol_reg_other.predict(poly_reg_non.fit_transform(ukdate))

plt.scatter(ukdate, X_other)
plt.plot(ukdate, linear_pred_other, '+-', color='black')
plt.plot(ukdate, poly_pred_other, '+-', color='red')
plt.title("Other stores")
plt.xlabel('Date')
plt.ylabel("Other stores")
plt.legend(['linear_reg', 'polynomial_reg', 'data'])
plt.show()


# In[134]:


print('Score:', mean_squared_error(X_other, linear_pred_other))
print('Score:', mean_squared_error(X_other, poly_pred_other))
print('Linear Model Fit Score:', lin_reg_other.score(ukdate, X_other))
print('Polynomial Model Fit Score:', pol_reg_other.score(X_poly_other, X_other))


# In[135]:


print('Linear RegressionOther stores Predictions')
print('20 Nov: ', lin_reg_other.predict([[131]]))
print('20 Dec: ', lin_reg_other.predict([[132]]))
print('21 Jan: ', lin_reg_other.predict([[133]]))
print('21 Feb: ', lin_reg_other.predict([[134]]))
print('21 Mar: ', lin_reg_other.predict([[135]]))


# In[136]:


print('Polynomial Regression Other stores Predictions')
print('20 Nov: ', pol_reg_other.predict(poly_reg_other.fit_transform([[131]])))
print('20 Dec: ', pol_reg_other.predict(poly_reg_other.fit_transform([[132]])))
print('21 Jan: ', pol_reg_other.predict(poly_reg_other.fit_transform([[133]])))
print('21 Feb:  ', pol_reg_other.predict(poly_reg_other.fit_transform([[134]])))
print('21 Mar:  ', pol_reg_other.predict(poly_reg_other.fit_transform([[135]])))


# ### Real Results
# 
# 20-Oct	19.9
# 
# 20-Nov	27.8
# 
# 20-Dec	22.9
# 
# 21-Jan	39.1
# 
# 21-Feb	40.1
# 
# 21-Mar	35.6
# 
# 

# As a conclusion we can't reject H0. The rates went up in each category.
