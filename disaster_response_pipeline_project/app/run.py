import json
import plotly
import pandas as pd
import re
import string
from nltk.corpus import stopwords




import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
# from sklearn.externals import joblib

from sqlalchemy import create_engine


app = Flask(__name__)





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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# extract data needed for visuals
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

df_categories = df.drop(['id'], axis=1)._get_numeric_data()
top_categories_pcts = df_categories.sum().sort_values(ascending=False).head(10)
top_categories_names = list(top_categories_pcts.index)

words = df.message.str.cat(sep=' ').lower().translate(str.maketrans('', '', string.punctuation)).split()
df_words = pd.Series(words)
top_words = df_words[~df_words.isin(stopwords.words("english"))].value_counts().head(10)
top_words_names = list(top_words.index)




# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words_names,
                    y=top_words
                )
            ],

            'layout': {
                'title': 'Top 10 Message Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_categories_names,
                    y=top_categories_pcts
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    ####################################

#     ####### VARIABLE CALCULATION #######
#     messages = df['message']
#     carac_raw = list()
#     tokens_raw = list()
#     tokens_proces = list()

#     for i in range(len(messages)):

#         #caracteres por descrição
#         carac_raw.append(len(messages[i]))

#         # tokens eh uma lista de strings
#         tokens = nltk.word_tokenize(messages[i])

#         tokens_raw.append(len(tokens))

#         # words contem so palavras, siglas. Nao inclui pontuação
#         words = [word for word in tokens if word.isalpha()]

#         # contem o numero de palavras por descricao 
#         n_palavras = len(words)

#         # proporcao de palavras entre os tokens
#         try:
#             prop_palavras = n_palavras/len(tokens)
#         except:
#             prop_palavras = 0

#         # words_len tem o tamanho de cada palavra
#         words_len = [len(worddd) for worddd in words]

        
#     fig = px.histogram(carac_raw,nbins=20)
#     fig.show()
    
    
#     graphs2 = [ 
#         {
#             'data': [
#                 Bar(
#                     x=genre_names,
#                     y=genre_counts
#                 )
#             ],

#             'layout': {
#                 'title': 'Distribution of Message types',
#                 'yaxis': {
#                     'title': "Count"
#                 },
#                 'xaxis': {
#                     'title': "Genre"
#                 }
#             }
#         }
#     ]
    
#     # encode plotly graphs in JSON
#     ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
#     graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)




#     fig = px.histogram(carac_raw,nbins=20)
#     fig.show()
    
    #################################
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

    
#     Here are the steps:
# 1. Run the web app inside the 5_deployment folder. In the terminal, use this command to get the link for vieweing the app:
# env | grep WORK

# The link wil be:
# http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

# To run the web app, go into the Terminal and type:
# python app/run.py

## http://view6914b2f4-3001.udacity-student-workspaces.com

## message example: I would like to receive the messages, thank you