import pandas as pd
from sklearn import linear_model,preprocessing,model_selection
from matplotlib import pyplot as plt
import seaborn as sns


df=pd.read_csv("day.csv")
#df.head() 
df.dropna() 
#print(df["season"].unique())

#Considering every non-working day is a holiday (dropping weekday column), if its a non-working day, then holiday= 1
mask=df.workingday=0
column_name='holiday'
df.loc[mask,column_name]=1

#Defining feature dataset. Dropping the columns that do not affect the value of count of bikes booked on a particular day and the target column.
X=df.drop(df[['instant','dteday','mnth','weekday','workingday','atemp','windspeed','casual','registered','cnt']],axis=1)

X['weathersit']=X['weathersit'].map({1:'000', 2:'001', 3:'010'}) 
X['yr']=X['yr'].map({0:'00', 1:'01'})
X['season']=X['season'].map({1:'000', 2:'001', 3:'010', 4: '100'})

#Scaling temperature and humidity variables
X[['temp','hum']]=preprocessing.scale(X[['temp','hum']]) 

print(X) 

#Defining target dataset.
Y=df['cnt']
print(Y)


#Variation of count with the change in season
plt.figure(1)
plt.subplot(1,2,1)
left = df['season'].unique()
height=df['season'].value_counts()
tick_label = ['Fall', 'Summer', 'Spring', 'Winter']
plt.bar(left, height, tick_label=tick_label, color='violet')
plt.xlabel('Season') 
plt.ylabel('Cnt')
plt.title('Change in season vs Cnt')


#Variation of count depending on whether its a working day or holiday
plt.subplot(1,2,2)
left = df['holiday'].unique()
height=df['holiday'].value_counts()
tick_label = ['Yes', 'No']
plt.bar(left, height, tick_label=tick_label, color='violet')
plt.xlabel('Holiday') 
plt.ylabel('Cnt')
plt.title('Cnt on workday & holiday')
plt.tight_layout()


#Variation of count with the change in weather
plt.figure(2)
plt.subplot(1,2,1)
left = df['weathersit'].unique()
height=df['weathersit'].value_counts()
tick_label = ['Clear', 'Mist','Light Rain']
plt.bar(left, height,tick_label=tick_label,  color='violet')
plt.xlabel('Weather Situation') 
plt.ylabel('Cnt')
plt.title('Change in weather vs Cnt')


#Plot of count for the year 2011 and 2012
plt.subplot(1,2,2)
left=df['yr'].unique()
height=df['yr'].value_counts()
tick_label=['2011','2012']
plt.bar(left,height,tick_label=tick_label, color='violet')
plt.xlabel('Year') 
plt.ylabel('Cnt')
plt.title('Cnt for 2011 & 2012')


plt.tight_layout()
#plt.show()

#Univariate distribution of temperature
plt.figure(3)
plt.subplot(1,2,1)
sns.distplot(X['temp'])

plt.subplot(1,2,2)
X['temp'].plot.box()

#Univariate distribution of count of bikes booked
plt.figure(4)
plt.subplot(1,2,1)
sns.distplot(df['cnt'])

plt.subplot(1,2,2)
df['cnt'].plot.box()
plt.tight_layout()


#Create linear regression object
lr=linear_model.LinearRegression()

#Dividing data into train and test with 4:1 ratio and train the model using the training sets.
X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X,Y,test_size=0.2, random_state=6)
lr.fit(X_train,Y_train)

#Regression coefficients 
print('Coefficients: \n', lr.coef_)

#Variance score
print('Variance score: {}'.format(lr.score(X_test,Y_test)))


#Plotting residual errors in training data 
plt.scatter(lr.predict(X_train), (lr.predict(X_train)-Y_train),s=10,color='green', label = 'Train data')

#Plotting residual errors in test data 
plt.scatter(lr.predict(X_test), (lr.predict(X_test)-Y_test),s=10, color='blue', label = 'Test data')

#Plotting line for zero residual error
plt.hlines(xmin=0,xmax=9000,y=0,linewidth=2, color='black')

plt.legend(loc = 'upper right') 
plt.title("Residual errors")

