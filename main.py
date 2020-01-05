#%% Import packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#%% Import Datasets
train_df = pd.read_csv('../input_data/train.csv')
test_df = pd.read_csv('../input_data/test.csv')
submission_df = pd.read_csv('../input_data/sample_submission.csv')

#%% Define variables for replacing to get cleaned text
sr = stopwords.words('english')
short_forms = ["i'm", "you're"]
punctuations = ["'", "!", "#", "."]

#%% Convert to lower characters
train_df['text_processed'] = train_df['text'].str.lower()

# Removing couple of short forms
train_df['text_processed'] = train_df['text_processed'].apply(lambda x: \
  ' '.join([y if y not in short_forms else '' for y in x.split(' ')]))
    
# Removing stopwords
train_df['text_processed'] = train_df['text_processed'].apply(lambda x: \
  ' '.join([y if y not in sr else '' for y in x.split(' ')]))

# Removing punctuations
for symbol in punctuations:
    train_df['text_processed'] = \
        train_df['text_processed'].str.replace(symbol, '')
        
# Removing leading and trailing spaces
train_df['text_processed'] = train_df['text_processed'].str.strip()

#%% Vectorizing for 
vectorizer = CountVectorizer()

X = train_df['processed_text']



