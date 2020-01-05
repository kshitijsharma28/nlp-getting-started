#%% Import packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB

#%% Import Datasets
train_df = pd.read_csv('../input_data/train.csv')
test_df = pd.read_csv('../input_data/test.csv')
submission_df = pd.read_csv('../input_data/sample_submission.csv')

#%% 
def text_cleanup(text_col):
    # Define variables for replacing to get cleaned text
    sr = stopwords.words('english')
    short_forms = ["i'm", "you're"]
    punctuations = ["'", "!", "#", "."]

    # Convert to lower characters
    output_col = text_col.str.lower()
    
    # Removing couple of short forms
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in short_forms else '' for y in x.split(' ')]))
        
    # Removing stopwords
    output_col = output_col.apply(lambda x: \
      ' '.join([y if y not in sr else '' for y in x.split(' ')]))
    
    # Removing punctuations
    for symbol in punctuations:
        output_col = output_col.str.replace(symbol, '')
            
    # Removing leading and trailing spaces
    output_col = output_col.str.strip()
    
    return output_col

#%%
train_df['text_processed'] = text_cleanup(train_df['text'])

#%% Vectorizing for 
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=33)
 
    classifier.fit(X_train, y_train)
    model_performance(y_test, classifier.predict(X_test))
    return classifier

#%%
trial1 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB()),
])
 
clf = train(trial1, train_df['text_processed'], train_df['target'])
# %%
test_df['target'] = clf.predict(text_cleanup(test_df['text']))
test_df[['id', 'target']].to_csv('../submissions/submission_naivebayes.csv', \
                                 index=False)
#%%

    
    
    
    
def model_performance(y_true, y_pred):
    print('%12s%12s%12s' % ('Accuracy', 'Precision', 'Recall'))
    print('%s' % '-'*30)
    print('%12.5f %12.5f %12.5f' % (accuracy_score(y_true, y_pred), \
                                    precision_score(y_true, y_pred), \
                                    recall_score(y_true, y_pred)))