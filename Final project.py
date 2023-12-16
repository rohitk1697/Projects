#AUTHOR:ROHIT KHILARE
#FINAL PROJECT : PUBLIC PERCEPTION ON MASS SHOOTING
#DATE: 12/03/2023



#Lets start by importing required libraries

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords

#Loading the CSV file with mass shooting data provided
mass_shootings_df = pd.read_csv('data-project.csv')

#Checking for missing values in csv file
missing_values = mass_shootings_df.isnull().sum()

#Handling missing value by dropping columns having data missing more than 20%
threshold = 0.2 * len(mass_shootings_df)
mass_shootings_df = mass_shootings_df.dropna(thresh=threshold, axis=1)

#Loading text files from Epoch , CNN , NY times
with open('CNN.txt', 'r', encoding='utf-8') as file:
    cnn_text = file.read()

with open('Epoch.txt', 'r', encoding='utf-8') as file:
    epoch_text = file.read()

with open('New York Times.txt', 'r', encoding='utf-8') as file:
    nytimes_text = file.read()


    plt.figure(figsize=(8, 6))                                                          #Histogram of the number of casualties
    sns.histplot(mass_shootings_df['fatalities'], bins=20, kde=True)
    plt.title('Distribution of Total Casualties in Mass Shootings')
    plt.xlabel('Total Casualties')
    plt.ylabel('Frequency')
    plt.show()

                                                                                        #Boxplot of casualties by year
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=mass_shootings_df['year'], y=mass_shootings_df['fatalities'])
    plt.title('Boxplot of Total Casualties by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Casualties')
    plt.show()

                                                                                    #Total number of mass shootings each year
shootings_by_year = mass_shootings_df.groupby('year')['fatalities'].count()

                                                                                    #Ploting the trend
plt.figure(figsize=(12, 8))
sns.lineplot(x=shootings_by_year.index, y=shootings_by_year.values, marker='o')
plt.title('Trend in Number of Mass Shootings (2010-2023)')
plt.xlabel('Year')
plt.ylabel('Number of Mass Shootings')
plt.show()


                                                                                     #correlation matrix
columns_of_interest = ['fatalities', 'injured', 'total_victims', 'age']

                                                                                    #Creating a copy of the DataFrame to avoid the SettingWithCopyWarning
correlation_data = mass_shootings_df[columns_of_interest].copy()

                                                                                    #Replace non-numeric values ('-') with NaN as there was queit some data in csv file
correlation_data.replace('-', np.nan, inplace=True)

                                                                                    #Converting columns to numeric
correlation_data = correlation_data.apply(pd.to_numeric, errors='coerce')

                                                                                    #Calculating correlation matrix
correlation_matrix = correlation_data.corr()

                                                                                    #Ploting heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

#Function for Sentiment analysis
def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

                                                                                    #Applying sentiment analysis to each source's text file
cnn_sentiment = get_sentiment(cnn_text)
epoch_sentiment = get_sentiment(epoch_text)
nytimes_sentiment = get_sentiment(nytimes_text)

                                                                                    #Visualizing sentiment scores
labels = ['CNN', 'Epoch', 'NY Times']
sentiments = [cnn_sentiment, epoch_sentiment, nytimes_sentiment]

plt.bar(labels, sentiments, color=['blue', 'green', 'red'])
plt.title('Sentiment Analysis of News Sources on Mass Shootings')
plt.ylabel('Sentiment Score')
plt.show()




                                                                                    #Analyzing trends in mass shootings
mass_shootings_df['Date'] = pd.to_datetime(mass_shootings_df['year'])
mass_shootings_df.set_index('Date', inplace=True)

                                                                                    #Resampling the data to analyze trends over time
monthly_shootings = mass_shootings_df.resample('M').size()

                                                                                    #Visualize the trend in the number of mass shootings
plt.plot(monthly_shootings.index, monthly_shootings.values, marker='o')
plt.title('Trend in Monthly Mass Shootings (2010-2023)')
plt.xlabel('Date')
plt.ylabel('Number of Shootings')
plt.show()

# Function to generate word cloud for press opinion and public percertion
stop_words = set(stopwords.words('english'))
#Loading stopwords
stop_words = set(stopwords.words('english'))

# Function to generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

#Loading text files
with open('CNN.txt', 'r', encoding='utf-8') as file:
    cnn_text = file.read()

with open('Epoch.txt', 'r', encoding='utf-8') as file:
    epoch_text = file.read()

with open('New York Times.txt', 'r', encoding='utf-8') as file:
    nytimes_text = file.read()

                                                                                                #Generating word clouds
generate_wordcloud(cnn_text, 'CNN Word Cloud')
generate_wordcloud(epoch_text, 'Epoch Word Cloud')
generate_wordcloud(nytimes_text, 'NY Times Word Cloud')


#Geographical distribution using a bar plot or map
plt.figure(figsize=(12, 6))
sns.countplot(x='location', data=mass_shootings_df)
plt.title('Geographical Distribution of Mass Shootings')
plt.xlabel('State')
plt.ylabel('Number of Mass Shootings')
plt.xticks(rotation=45)
plt.show()

#Weapon analysis
weapons_used = mass_shootings_df['weapon_details'].value_counts()

# Ploting the types of weapons used
plt.figure(figsize=(15, 10))
sns.barplot(x=weapons_used.index, y=weapons_used.values)
plt.title('Types of Weapons Used in Mass Shootings')
plt.xlabel('Weapon Type')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

#Victims analysis

plt.figure(figsize=(10, 6))
sns.boxplot(x='total_victims', y='fatalities', data=mass_shootings_df)
plt.title('Distribution of Total Casualties by News Source')
plt.xlabel('Total Casualties')
plt.ylabel('News Source')
plt.show()

#Displaying basic statistics

basic_stats = mass_shootings_df.describe()
print(basic_stats)
