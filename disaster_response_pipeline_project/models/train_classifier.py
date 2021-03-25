import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def load_data(database_filepath):

    engine = create_engine('sqlite:///'+ database_filepath) #nome do arquivo
    df = pd.read_sql_table("merged_df", engine) #nome da tabela


    X = df['message']
    Y = df.iloc[:,4:40]
    category_names= [Y.columns]

    return X, Y, category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

        
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.wordpunct_tokenize(text)
    text = nltk.Text(tokens)

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in text if w.isalpha()]


    return clean_tokens


def build_model():# I need some help and feedback here
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        # 'vect__max_df': (0.5, 0.75),
        # 'vect__max_features': (1000, 3000),
        # 'vect__ngram_range': ((1,1),(1,2)) ,
        # 'tfidf__use_idf': (True,False),
        'clf__estimator__n_neighbors': (3,5),
        # 'clf__estimator__weights': ('uniform','distance'),
        # 'clf__estimator__metric': ('minkowski','euclidean')
    
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    for i,col in enumerate(Y_test.columns):
        print(classification_report(Y_test[col], y_pred[:,i]))

    pass

def save_model(model, model_filepath):

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


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

# estando em Disaster-Response-Pipeline
# git lfs track disaster_response_pipeline_project/models/classifier.pkl
# python disaster_response_pipeline_project//models//train_classifier.py DataBase.db disaster_response_pipeline_project//models//classifier.pkl

# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl