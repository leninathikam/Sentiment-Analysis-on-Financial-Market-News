
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

knn=KNeighborsClassifier()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
lr=LogisticRegression()
mb=MultinomialNB()


df=pd.read_csv(r"financial_data.csv")

df #checking dataset

df.head()#checking forst 5 values

df["neutral"].value_counts()

df.rename(columns ={'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'text'},inplace=True)
#We are changing the name of the column.

df["text"]=df["text"].str.lower() #We convert our texts to lowercase.
df["text"]=df["text"].str.replace("[^\w\s]","") #We remove punctuation marks from our texts.
df["text"]=df["text"].str.replace("\d+","") #We are removing numbers from our texts.
df["text"]=df["text"].str.replace("\n","").replace("\r","") #We remove spaces in our texts.
df_neutral=df[df['neutral']=="neutral"]
df_positive=df[df['neutral']=="positive"]
df_negative=df[df['neutral']=="negative"]
df["neutral"]=df["neutral"].map({"positive":1,"negative":0,"neutral":2})
df['neutral']=df['neutral'].astype(int)
df1=df[df['neutral']!=2]
#We divide it into positive and negative.

vect=CountVectorizer(lowercase=True,stop_words="english")
x=df1.text
y=df1.neutral
x=vect.fit_transform(x)

def sentiment_classification_funct(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)

    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

    knn=KNeighborsClassifier()
    dt=DecisionTreeClassifier()
    rf=RandomForestClassifier()
    lr=LogisticRegression()
    mb=MultinomialNB()

    algos=[knn,dt,rf,lg,mb]
    algo_names=['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression','MultinomialNB']

    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]

    for item in algos:
        item.fit(x_train,y_train)
        accuracy_scored.append(accuracy_score(y_test,item.predict(x_test)))
        precision_scored.append(precision_score(y_test,item.predict(x_test)))
        recall_scored.append(recall_score(y_test,item.predict(x_test)))
        f1_scored.append(f1_score(y_test,item.predict(x_test)))

    result=pd.DataFrame(columns=['f1_score','recall_score','precision_score','accuracy_score'],index=algo_names)
    result.f1_score=f1_scored
    result.recall_score=recall_scored
    result.precision_score=precision_scored
    result.accuracy_score=accuracy_scored
    sentiment_classification_funct.result=result.sort_values('f1_score',ascending=False)
    return result.sort_values('f1_score',ascending=False)

sentiment_classification_funct(x,y)

def wc(data,bgcolor):
    plt.figure(figsize=(10,10))
    #mask=np.array(Image.open("Stock Market.png"))
    wc=WordCloud(background_color=bgcolor,stopwords=STOPWORDS,mask=mask)
    wc.generate(" ".join(data))
    plt.imshow(wc)
    plt.axis("off")
#We draw the most used words in texts on a picture.

wc(df_positive.text,"black")##Positive

wc(df_negative.text,"black")##Negative

wc(df_neutral.text,"black")##Neutral

sent=df[["neutral","text"]]

def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity
#We are doing our sentiment analysis.

sent["sentiment"]=sent["text"].apply(detect_sentiment)
sent.head()

def sentiment2(sent):
    if (sent< -0.02):
        return 3
    elif sent>0.02:
        return 1
    else:
        return 0
#We divide the texts into three groups positive, negative and nötr.

sent["sent"]=sent["sentiment"].apply(sentiment2)
sent.head()

sent.sentiment.value_counts()

