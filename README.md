# <center>SENTIMENT ANALYSIS
## A Case Study of Talabat Company


## About Talabat
Talabat is an online food ordering company founded in Kuwait in 2004. As of April 2021, Talabat operates in Kuwait, Saudi Arabia, Bahrain, the United Arab Emirates, Oman, Qatar, Jordan, Egypt, and Iraq.

## About The Dataset
### Context
This dataset contains information scraped from twitter  for the period June to November 2022. I used python script to scrape tweets with the keyword “Talabat”.  The tweets were saved to CSV in 3 progressive files. A total of 9,006 tweets were collected and processed.

This is what the original CSV dataset looks like after scraping and formatting into CSV row-column:
![image](https://user-images.githubusercontent.com/69392408/206904458-a0d37cea-b4e3-450e-a311-72571afec063.png)

 

## Objective Of The Analysis
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a negative sentiment associated with it. So, the task is to classify negative, or positive tweets from other tweets based on how twitter users feel about Talabat as a company and it’s services.

## Data Analytic Tool
Python programming language was used in this project, putting into use the Natural language processing technique among other features.

The needed modules were imported and used. These includes:
```
##Import modules
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
from nltk.stem.porter import *
from wordcloud import WordCloud
```

## Data Preparations & Processing

The 3 CSV files created from the twitter scrapping were named “talabat.csv”, “talabat1.csv”,  and “talabat2.csv”.  The 3 files were imported into dataframe on python for processing to begin.
```
##Import dataset
data = pd.read_csv('talabat.csv')
data1 = pd.read_csv('talabat1.csv')
data2 = pd.read_csv('talabat2.csv')
```
Then, they were merged into a single dataset called “talabat_data” and then into a single csv file.

```
##Merge the datasets
talabat_temp = pd.concat((data,data1), axis=0)
talabat_data = pd.concat((talabat_temp,data2), axis=0)

talabat_data.to_csv('talabat_data.csv', encoding='UTF-8')
```
```
#Inspecting the dataset
talabat_data.head(30)
```
![image](https://user-images.githubusercontent.com/69392408/206904439-688b34cf-c365-41d1-b461-bdfc98c64046.png)

Initial data cleaning requirements that we can think of after looking at the top records:
*	The Twitter handles @user are hardly giving any information about the nature of the tweet.
*	We also need to get rid of the punctuations, numbers and even special characters.
*	Most of the smaller words do not add much value. For example, ‘and’, ‘his’, ‘all’. So, we will try to remove them as well from our data.
*	We can also split every tweet into individual words (tokenization), since it is an essential step in any NLP task.
*	We need to shrink related words. We have terms like loves, loving, lovable, etc. in the rest of the data. These terms are often used in the same context. If we can reduce them to their root word, which is ‘love’, then we can reduce the total number of unique words in our data without losing a significant amount of information.

```
##Function to remove pattern
def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text
```
```
# remove twitter handles
talabat_data['clean_tweet'] = np.vectorize(remove_pattern)(talabat_data['Text'], "@[\w]*")

#remove special characters, numbers and punctuations
talabat_data['clean_tweet'] = talabat_data['clean_tweet'].str.replace("[^a-zA-Z#]"," ")

#remove special characters, numbers and punctuations for location
talabat_data['clean_location'] = talabat_data['Location'].str.replace("[^a-zA-Z#]"," ")
talabat_data['clean_location'] = talabat_data['clean_location'].str.strip()
```
```
#Remove stop words
talabat_data['clean_tweet'] = talabat_data['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
```
```  
#talabat_data.loc[:,['Date', 'Text', 'clean_tweet']].head()
```
![image](https://user-images.githubusercontent.com/69392408/206904418-40b6bd03-d91f-4a84-b302-5c1390df76d4.png)

		
```
#Tokenize each tweet - splitting a string of texts into tokens/list
tokenize_tweets = talabat_data['clean_tweet'].apply(lambda x: x.split())
tokenize_tweets.head()
```
```  
#Stemming or shrinking related words in the tokenized list
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenize_tweets = tokenize_tweets.apply(lambda x: [stemmer.stem(i) for i in x])
tokenize_tweets.head()
```
0    [Hello, James, apologize, that, please, check,...
1                                                   []
2    [Hello, Taiba, apologize, experience, please, ...
3    [Hello, apologize, that, please, send, with, y...
4                                                   []

Name: clean_tweet, dtype: object

```
#join the token back together
for i in range(len(tokenize_tweets)):
    tokenize_tweets[i] = ' '.join(tokenize_tweets[i])

talabat_data['clean_tweet'] = tokenize_tweets
```

#before
21    [Food, delivery, aggregators, like, should, ha...
22    [Thank, Kuwait, helping, clean, beaches, prote...
23    [Mohammed, Sorry, hear, about, this, have, fol...

#after
21    [food, deliveri, aggreg, like, should, have, o...
22    [thank, kuwait, help, clean, beach, protect, m...
23     [moham, sorri, hear, about, thi, have, follo





```
talabat_data.head(30)
```       
![image](https://user-images.githubusercontent.com/69392408/206925659-a48f2a14-930e-46f6-8111-01941e8004c9.png)


## Visualization From Tweets Dataset
We will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. 
A few probable questions questions related to the data in hand are:
*	What are the most common words in the entire dataset?
*	Does the most common words in the dataset reflect negative or positive tweets about Talabat?
*	What are the most used hashtags in these tweets and what do they reflect?

**Let’s discover the common words used in the tweets via WordCloud**
A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.

```
#word cloud generation
all_words = ' '.join([text for text in talabat_data['clean_tweet']])

from wordcloud import WordCloud

wc = WordCloud(width=750, height=450, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10,7))

plt.imshow(wc, interpolation="bilinear")

plt.title("Talabat Tweets Word Cloud")

plt.axis("off")

plt.show()
```
![image](https://user-images.githubusercontent.com/69392408/206904259-350abe27-d01c-455c-a413-35089e3bf927.png)

### Observation:
The most frequently used words are the biggest in size, and these are neither positive nor negative words. The are neutral. Next to these set of neutral words are positive and service related words, like “good”, “delivery”, ”customer service”. 

This gives us hint that most tweeter users are in good standing with Talabat service across the region. 


**Let’s explore the impact of Hashtags on tweets sentiment**
	
Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular point in time. 
We will try to check whether these hashtags add any value to our sentiment analysis task, if they will help in distinguishing tweets into the different sentiments.

```
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from all tweets
HT_all = hashtag_extract(talabat_data['clean_tweet'])

# unnesting list
HT_all = sum(HT_all,[])

a = nltk.FreqDist(HT_all)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})


# selecting top 15 most frequent hashtags     
d = d.nlargest(columns="Count", n = 15) 
plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Count", y = "Hashtag", orientation='horizontal')

ax.set(title = 'Talabat Tweets Hashtags')

plt.show()
``` 
![image](https://user-images.githubusercontent.com/69392408/206904237-56df6b6f-6203-4550-bdfe-02df423e98e7.png)

### Observation:
Once again, the chart shows us that most hashtags used are neutral: location based and service based.


**Finally, let us get the polarity from cleaned tweets**
Sentiment polarity for an element defines the orientation of the expressed sentiment. It determines if the text expresses the positive, negative or neutral sentiment of the user about the entity in consideration, in our case, “TALABAT”.

```
# Polarity score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# adding a row_id field to the dataframe, which will be useful for joining dataframes later

talabat_data["row_id"] = talabat_data.index + 1

#create a new data frame with "id" and "comment" fields
df_subset = talabat_data[['row_id', 'Text']].copy()

#covert to lower-case
df_subset['clean_tweet'] = df_subset['clean_tweet'].str.casefold()


# set up empty dataframe for staging output
df1=pd.DataFrame()
df1['row_id']=['99999999999']
df1['sentiment_type']='NA999NA'
df1['sentiment_score']=0

print('Processing sentiment analysis...')
sid = SentimentIntensityAnalyzer()
t_df = df1
for index, row in df_subset.iterrows():
    scores = sid.polarity_scores(row[1])
    for key, value in scores.items():
        temp = [key,value,row[0]]
        df1['row_id']=row[0]
        df1['sentiment_type']=key
        df1['sentiment_score']=value
        t_df=t_df.append(df1)

#remove dummy row with row_id = 99999999999
t_df_cleaned = t_df[t_df.row_id != '99999999999']

#remove duplicates if any exist
t_df_cleaned = t_df_cleaned.drop_duplicates()

# only keep rows where sentiment_type = compound
t_df_cleaned = t_df[t_df.sentiment_type == 'compound']

df_output = talabat_data.merge(t_df_cleaned, on='row_id', how='inner')

df_output[["sentiment_score"]].describe()
df_output1 = df_output['clean_location'].nlargest(10)

#generate mean of sentiment_score by period
dfg = df_output.groupby(['clean_location'])['sentiment_score'].mean()

#create a bar plot
plt.xlabel('Location')
plt.ylabel('Mean Sentiment Score')
dfg.plot(kind='bar', title='Sentiment Score', y='sentiment_score', x='clean_location', figsize=(6, 5))
plt.show()
```

![image](https://user-images.githubusercontent.com/69392408/206904612-0299d2fd-dd2e-4d3d-8996-faa16b9bf2fa.png)


```
df_output_mnt =df_output

#### show for date-month grouping
#generate mean of sentiment_score by period
dfg = df_output_mnt.groupby(['mnt'])['sentiment_score'].mean()

#create a bar plot
plt.xlabel('Period of Tweet')
plt.ylabel('Mean Sentiment Score')
dfg.plot(kind='bar', title='Sentiment Score', y='sentiment_score', x='Location', figsize=(6, 5))
plt.show()
```

![image](https://user-images.githubusercontent.com/69392408/206925769-323b11b2-349e-4d7a-8815-bbd952d0ae5a.png)
 

## Conclusion

Generally, Talabat company is perceived by most twitter users sampled as a good company to deal with, and interestingly, there are no negative tweets across the different countries in which Talabat operates.
 
