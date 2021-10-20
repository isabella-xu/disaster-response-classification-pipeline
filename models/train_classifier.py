import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    This fuction is to load data from database.
    
    INPUT:
    - database_file: file path where database was saved
    
    OUTPUT:
    - X: training message List
    - Y: training target
    - category_names: category names for labeling
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('FigureEight', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    
    return X, Y, category_names

def tokenize(text):
    '''
    This function is to tokenize messages
    
    INPUT:
    - text: message data
    
    OUTPUT:
    - clean_tokens: tokens extracted from messages
    '''
    # Define regex to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder:
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize the text
    text = re.sub(r'[^A-Za-z0-9]', " ", text.lower())
    words = word_tokenize(text)
    
    # remove stopwords
    tokens = [ele for ele in words if ele not in stopwords.words('english')] 
    
    # lemmatize verbs by specifying pos
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    This function specifies the pipeline and the grid search parameters to build a
    classification model.
    
    INPUT: None
    
    OUTPUT:
    - cv: classification model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
    ])
    
    # Choose parameters:
    parameters = {
     'clf__estimator__n_estimators': [5]
    }
    
    # Apply GridSearchCV model:
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function is to evaluate model prediction.
    
    INPUT: 
    - model: 
    - X_test: test data
    - Y_test: test target data
    - category_names: a list of category names
    
    OUTPUT:
    - cv: classification model
    '''

    # predict with test data
    y_pred = model.predict(X_test)
    
    threshold = 0.5
    
    # Check f1 scores.
    print(classification_report(Y_test, y_pred > threshold, target_names = category_names))
    print('-----------------------------------------------------')
    for i in range(Y_test.shape[1]):
        print('%25s accuracy report: %.2f' %(category_names[i], accuracy_score(Y_test[:,i], y_pred[:,i])))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()