import GetOldTweets3 as got
import pandas as pd
import numpy as np
from textblob import TextBlob
import string
import re
import streamlit as st
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from pycaret.nlp import *
import spacy
nlp = spacy.load("en_core_web_sm")


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocessing(phrase):
    # we are removing the words from the stop words list: 'no', 'nor', 'not'
    custom_stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(custom_stopwords)
    
 
    stemmer = nltk.stem.SnowballStemmer("english", ignore_stopwords=False)
    preprocessed_reviews = []
    # tqdm is for printing the status bar
    for sentance in phrase.values:
        sentance=re.sub('@[\w]*','',sentance)
        sentance=re.sub('#[\w]*','',sentance)
        sentance=re.sub('pic[\w]*','',sentance)
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        sentance = sentance.lower()
        sentance = [stemmer.stem(word) for word in sentance.split() if word not in stopwords]
        sentance = ' '.join(sentance)
    
        preprocessed_reviews.append(sentance.strip())
    final_df=pd.DataFrame(preprocessed_reviews)
    return preprocessed_reviews

def get_sentiment(review):
    senti = SentimentIntensityAnalyzer()
    score = senti.polarity_scores(review)['compound']
    if score > 0.1:
        sentiment = 'Positive'
    elif score < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment

def plots(input_df):
    g = sns.countplot(x = 'Dominant_Topic' , data = input_df )
    plt.title('Topic modelling')
    for p in g.patches:
        g.annotate(format(p.get_height() ), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    st.pyplot()
        
    input_df['Type of sentiment'] = input_df['text'].apply(get_sentiment)
    g = sns.countplot(x = 'Type of sentiment' , data = input_df )
    plt.title('Sentiment analysis')
    for p in g.patches:
        g.annotate(format(p.get_height() ), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    st.pyplot()
        
    for i in input_df['Dominant_Topic'].unique():
        df1 = input_df[input_df['Dominant_Topic']==i]
            
        wc = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join( df1['text']))
        plt.figure(figsize=(15,15))
        plt.axis('off')
        plt.title('Word cloud of' + ' ' + i)
        plt.imshow(wc)
        plt.show()
        st.pyplot()

def run():
    
    
    from PIL import Image
    image = Image.open('twitter.png')
    image2 = Image.open('sm.jpg')
    st.image(image , use_column_width = False)
    st.title('Online Tweet Analyzing App')
    st.sidebar.subheader(' Created By Saideepak')
    st.sidebar.image(image2, use_column_width = False)

    
    select_option = st.sidebar.selectbox(
    "Choose a topic modelling algorithm",
    ('Latent Dirichlet Allocation', 'Latent Semantic Indexing', 'Hierarchical Dirichlet Process','Random Projections',
    'Non-Negative Matrix Factorization'))
    
    search_term = st.text_input('Search term')
    start_date = st.text_input('Start date (yyyy-mm-dd)')
    end_date = st.text_input('End date (yyyy-mm-dd)')
    max_tweets = st.number_input('Maximum number of tweets', min_value = 1)
    no_topics = st.number_input('Number of topics', min_value = 2)
    
    if st.button('Analyze'):
        df=pd.DataFrame()
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch( search_term + '-filter:retweets -filter:replies')\
                                           .setSince(start_date)\
                                           .setUntil(end_date).setMaxTweets(max_tweets)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        for tweet in tweets:
            curr_tweet = {
            'text' : tweet.text,
            'created_at' : tweet.date,
            #'username' : tweet.username,
            #'user_handler' : tweet.user.screen_name,
            'retweets': tweet.retweets,
            'likes': tweet.favorites,
            'hashtags':tweet.hashtags,
            #'mention':tweet.mentions,
            #'follower_count': tweet.user.followers_count 
            #'location': tweet.geo
        }
            df = df.append(curr_tweet, ignore_index=True)
            
            
        doc = preprocessing(df['text'])
        new_doc = pd.DataFrame(doc)
        new_doc.columns = ['text']
        
     
        
        
        #intialize the setup
        exp_nlp = setup(data = new_doc , target = 'text' )
        
        if select_option == 'Latent Dirichlet Allocation':
            model1 = create_model(model = 'lda' , num_topics = no_topics )
            df = assign_model(model1)
            plotting = plots(df)
            plotting
        
        elif select_option == 'Latent Semantic Indexing':
            model1 = create_model(model = 'lsi' , num_topics = no_topics )
            df = assign_model(model1)
            plotting = plots(df)
            plotting
              
        elif select_option == 'Hierarchical Dirichlet Process':
            model1 = create_model(model = 'hdp' , num_topics = no_topics )
            df = assign_model(model1)
            plotting = plots(df)
            plotting
          
        elif select_option == 'Random Projections':
            model1 = create_model(model = 'rp' , num_topics = no_topics )
            df = assign_model(model1)
            plotting = plots(df)
            plotting
                    
        elif select_option == 'Non-Negative Matrix Factorization':
            model1 = create_model(model = 'nmf' , num_topics = no_topics )
            df = assign_model(model1)
            plotting = plots(df)
            plotting
            
    
 
        #plot_model(model1 , plot = 'frequency')
        #st.plotly_chart(plot1 , use_container_width = True )
               
        #st.write(df)   
if __name__=='__main__':
    run()