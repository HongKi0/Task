#import required libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#reading data files
store_df=pd.read_csv("store.csv")
train_df=pd.read_csv("train.csv")

store_df.head()
store_df.describe()
#Checking the no. of NaN vales
store_df.isna().sum()

train_df.head()
train_df.describe()
#Checking the no. of NaN values
train_df.isna().sum()

#Merging both the Dataframes into one based on the "Store" ID
df=store_df.merge(train_df,on=["Store"],how="inner")
df.head()
#(rowsxcolumns) of the merged DataFrame
df.shape
#Checking the no. of NaN values
df.isna().sum()
#Dropping columns
df=df.drop(columns=["PromoInterval","Promo2SinceWeek","Promo2SinceYear"])
#Handling NaN
df.CompetitionDistance.fillna(df.CompetitionDistance.mode()[0],inplace=True)
df.CompetitionOpenSinceMonth.fillna(1, inplace=True)
df.CompetitionOpenSinceYear.fillna(df.CompetitionOpenSinceYear.mode()[0], inplace=True)
df.CompetitionOpenSinceMonth=df.CompetitionOpenSinceMonth.astype(int)
df.CompetitionOpenSinceYear=df.CompetitionOpenSinceYear.astype(int)

#Find the range of data
plt.figure(figsize=(5,10))
sns.set(style="whitegrid")
sns.distplot(df["Sales"])
#Find the range of the data
plt.figure(figsize=(5,10))
sns.set(style="whitegrid")
sns.distplot(df["Customers"])
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxenplot(data=df,scale="linear",x="DayOfWeek",y="Sales",color="orange")
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
sns.boxenplot(y="Customers", x="DayOfWeek",data=df, scale="linear",color="orange")

#I cap off, the Customers at 3000, and Sales at 20,000.
df["Sales"]=df["Sales"].apply(lambda x: 20000 if x>20000 else x)
df["Customers"]=df["Customers"].apply(lambda y: 3000 if y>3000 else y)
print(max(df["Sales"]))
print(max(df["Customers"]))

df["Date"]=pd.to_datetime(df["Date"])
df["Year"]=df["Date"].dt.year
df["Month"]=df["Date"].dt.month
df["Day"]=df["Date"].dt.day
df["Week"]=df["Date"].dt.week%4
df["Season"] = np.where(df["Month"].isin([3,4]),"Spring",np.where(df["Month"].isin([5,6,7,8]), "Summer",np.where(df["Month"].isin ([9,10,11]),"Fall",np.where(df["Month"].isin ([12,1,2]),"Winter","None"))))

Holiday_Year_Month_Week_df=pd.DataFrame({"Holiday per week":df["SchoolHoliday"],"Week":df["Week"],"Month":df["Month"],"Year":df["Year"],"Date":df["Date"]})
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.drop_duplicates(subset=['Date'])
Holiday_Year_Month_Week_df=Holiday_Year_Month_Week_df.groupby(["Year","Month","Week"]).sum()
print(Holiday_Year_Month_Week_df)

df=df.merge(Holiday_Year_Month_Week_df, on=["Year","Month","Week"],how="inner")

customer_time_df=pd.DataFrame({"Avg CustomersPerMonth":df["Customers"],"Month":df["Month"]})
AvgCustomerperMonth=customer_time_df.groupby("Month").mean()
print(AvgCustomerperMonth)

customer_time_df=pd.DataFrame({"Avg CustomersPerWeek":df["Customers"],"Week":df["Week"],"Year":df["Year"],"Month":df["Month"]})
AvgCustomerperWeek=customer_time_df.groupby(["Year","Month","Week"]).mean()
print(AvgCustomerperWeek)

df=df.merge(AvgCustomerperMonth,on="Month",how="inner")
df=df.merge(AvgCustomerperWeek,on=["Year","Month","Week"],how="inner")
promo_time_df=pd.DataFrame({"PromoCountperWeek":df["Promo"],"Year":df["Year"],"Month":df["Month"],"Week":df["Week"],"Date":df["Date"]})
promo_time_df=promo_time_df.drop_duplicates(subset=['Date'])
promo_time_df=promo_time_df.groupby(["Year","Month","Week"]).sum()
print(promo_time_df)

df=df.merge(promo_time_df,on=["Year","Month","Week"], how="inner")
df=df.rename(columns={'CompetitionOpenSinceYear': 'year','CompetitionOpenSinceMonth':'month'})
df['CompetitionOpenSince'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
df=df.rename(columns={ 'year':'CompetitionOpenSinceYear','month':'CompetitionOpenSinceMonth'})

numerical_data_col=["Store","Competition Distance","Promo2","DayOfWeek","Sales","Customers","Open","SchoolHoliday","Year","Month","Day","Week"]
categorical_data_col=["StoreType","Assortment","Season"]
for i in categorical_data_col:
    p=0
    for j in df[i].unique():
        df[i]=np.where(df[i]==j,p,df[i])
        p=p+1

    df[i]=df[i].astype(int)
#The column StateHoliday contains 0,'0',a and b. This needs to be conerted to a pure numerical data column
df["StateHoliday"].unique()
df["StateHoliday"]=np.where(df["StateHoliday"] == '0' ,0,1)
df["StateHoliday"]=df["StateHoliday"].astype(int)

#EDA
plt.figure(figsize=(10,10))
sns.set(style="whitegrid",palette="pastel",color_codes=True)
sns.violinplot(x="DayOfWeek",y="Sales",hue="Promo",split=True, data=df)
plt.figure(figsize=(10,10))
sns.set(style="whitegrid",palette="pastel",color_codes=True)
sns.violinplot(x="DayOfWeek",y="Customers",hue="Promo",split=True, data=df)

plt.figure(figsize=(15,15))
sns.set(style="whitegrid")
df["CompetitionDistanceLOG"]=np.log(df["CompetitionDistance"])
sns.lineplot(x="CompetitionDistanceLOG", y="Sales", data=df)

sns.set(style="whitegrid")
g=sns.relplot(y="Avg CustomersPerWeek", x="Week", hue="Holiday per week", data=df)
g.fig.set_size_inches(10,10)

sns.set(style="whitegrid")
g=sns.relplot(y="Holiday per week", x="Week", hue="PromoCountperWeek", data=df)
g.fig.set_size_inches(10,10)

#Feature Engineering
#using public state holidays data from https://www.timeanddate.com/holidays/germany/2013
holid=df.loc[df.StateHoliday=='a']
bydate=df.groupby('Date')['Store'].count()
#number of stores celebrating holidays
bydate.head()

#Figuring out store locations based on state holidays
SN = holid.loc[holid.Date == '2013-11-20','Store'].values
print('{} stores located in Saxony.'.format(SN.shape[0]))
BW_BY_ST = holid.loc[holid.Date == '2013-01-06','Store'].values
print('{} stores located in BW, BY, ST.'.format(BW_BY_ST.shape[0]))
BW_BY_HE_NW_RP_SL = holid.loc[holid.Date == '2013-05-30','Store'].values
print('{} stores located in BW, BY, HE, NW, RP, SL.'.format(BW_BY_HE_NW_RP_SL.shape[0]))
BY_SL = holid.loc[holid.Date =='2013-08-15','Store'].values
print('{} stores located in BY,SL.'.format(BY_SL.shape[0]))
BB_MV_SN_ST_TH = holid.loc[holid.Date =='2013-10-31','Store'].values
print('{} stores located in BB, MV, SN, ST, TH.'.format(BB_MV_SN_ST_TH.shape[0]))
BW_BY_NW_RP_SL = holid.loc[holid.Date =='2013-11-01','Store'].values
print('{} stores located in BW, BY, NW, RP, SL.'.format(BW_BY_NW_RP_SL.shape[0]))
BW_BY = np.intersect1d(BW_BY_ST, BW_BY_HE_NW_RP_SL)
print('{} stores located in BW, BY.'.format(BW_BY.shape[0]))
ST = np.setxor1d(BW_BY_ST, BW_BY)
print('{} stores located in ST.'.format(ST.shape[0]))
BY = np.intersect1d(BW_BY, BY_SL)
print('{} stores located in BY.'.format(BY.shape[0]))
SL = np.setxor1d(BY, BY_SL)
print('{} stores located in SL.'.format(SL.shape[0]))
BW = np.setxor1d(BW_BY, BY)
print('{} stores located in BW.'.format(BW.shape[0]))
HE = np.setxor1d(BW_BY_HE_NW_RP_SL,BW_BY_NW_RP_SL)
print('{} stores located in HE.'.format(HE.shape[0]))
BB_MV_TH = np.setxor1d(np.setxor1d(BB_MV_SN_ST_TH,SN),ST)
print('{} stores located in BB, MV, TH.'.format(BB_MV_TH.shape[0]))
NW_RP = np.setxor1d(BW_BY_NW_RP_SL,BW_BY) # SL has 0 stores
print('{} stores located in NW, RP.'.format(NW_RP.shape[0]))
allstores = np.unique(df.Store.values)
BE_HB_HH_NI_SH = np.setxor1d(np.setxor1d(allstores,BW_BY_HE_NW_RP_SL),BB_MV_SN_ST_TH)
print('{} stores located in BE, HB, HH, NI, SH.'.format(BE_HB_HH_NI_SH.shape[0]))

#using public school holidays data from http://www.holidays-info.com/School-Holidays-Germany/2015/school-holidays_2015.html.
#furthur division based on school holidays 
df.loc[df.Store.isin(NW_RP)].groupby('Date')['SchoolHoliday'].sum().value_counts()
RP = df.loc[df.Date=='2015-03-26'].loc[df.Store.isin(NW_RP)].loc[df.SchoolHoliday==1,'Store'].values
NW = np.setxor1d(NW_RP,RP)
print('{} stores located in RP.'.format(RP.shape[0]))
print('{} stores located in NW.'.format(NW.shape[0]))
df.loc[df.Store.isin(BB_MV_TH)].groupby('Date')['SchoolHoliday'].sum().value_counts()
TH = BB_MV_TH
print('{} stores located in TH.'.format(TH.shape[0]))
HH = df.loc[df.Date=='2015-03-02'].loc[df.Store.isin(BE_HB_HH_NI_SH)].loc[df.SchoolHoliday==1,'Store'].values
print('{} stores located in HH.'.format(HH.shape[0]))
BE_HB_NI_SH = np.setxor1d(BE_HB_HH_NI_SH,HH)
SH = df.loc[df.Date=='2015-04-17'].loc[df.Store.isin(BE_HB_NI_SH)].loc[df.SchoolHoliday==1,'Store'].values
print('{} stores located in SH.'.format(SH.shape[0]))
BE_HB_NI = np.setxor1d(BE_HB_NI_SH,SH)
BE = df.loc[df.Date=='2015-03-25'].loc[df.Store.isin(BE_HB_NI)].loc[df.SchoolHoliday==0,'Store'].values
print('{} stores located in BE.'.format(BE.shape[0]))
HB_NI = np.setxor1d(BE_HB_NI,BE)

states = pd.Series('',index = allstores,name='State')
states.loc[BW] = 'BW'
states.loc[BY] = 'BY'
states.loc[BE] = 'BE'
states.loc[HB_NI] = 'HB,NI'
states.loc[HH] = 'HH'
states.loc[HE] = 'HE'
states.loc[NW] = 'NW'
states.loc[RP] = 'RP'
states.loc[SN] = 'SN'
states.loc[ST] = 'ST'
states.loc[SH] = 'SH'
states.loc[TH] = 'TH'
states[states!=''].value_counts().sum()
states.to_csv('location.csv', header=True, index_label='Store')

location_df=pd.read_csv("./location.csv",index_col="Store")
location_df.head()

df=df.merge(location_df,on='Store',how="inner")

df.to_csv('final_RossmannSales.csv') 

df.head()
#Find Correlation between the data columns
plt.figure(figsize=(15,15))
sns.heatmap((df.corr()))
plt.title("Correlation Heatmap")
plt.show()