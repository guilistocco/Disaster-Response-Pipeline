# Disaster-Response-Pipeline
Web App repo with data and code with all steps of ETL process and deployment

Project to Udacity Nanodegree on Data Science. 
Started on Mar, 2021.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guilistocco/Disaster-Response-Pipeline/master)

 
 
## Why this is an important project?
In a Big Data context which we live in, there are much data, but not much knowledge. With this project I tried to show my skills on all steps of the ETL Process. It starts manipulating data from csv files, changing data types, removing duplicated values and ends saving all into a SQL database.

In adition, a web app is created to show some results obtained from the data. To do so Plotly makes data vizualizations and Flask along Boostrap make the app design.


## Summary of analysis
This project is capable of getting message and returning what it refers to in terms of themes as water shortage, huricanes and medical help.

Going deeper into analysis, the graphs show the distribution of tokens (mostly words) and caracters of each message of the training set 


## Libraries
* Pandas
* Numpy
* SQLAlchemy
* Flask
* Bootstrap
* Scikit Learn (a lot!)


## About the archives

* ETL-PROCESS folder: Jupyter Notebooks to create vizualizations, manipulate data and tests
* disaster_response_pipeline_project/app: files to run the app and html files
* disaster_response_pipeline_project/data: create and fill databases with clean/processed data
* disaster_response_pipeline_project/models: training files that takes around 20 min to run and save a KNN model with grid search


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Directory


Disaster-Response-Pipeline/
```
.
├── ETL-PROCESS/
│   ├── ETL-pipeline.ipynb
│   ├── Make-Viz.ipynb
│   └── Transform-Data.ipynb
├── disaster_response_pipeline_project/
│   ├── app/
│   │   ├── templates/
│   │   │   ├── Images/
│   │   │   │   ├── characters_per_message.png
│   │   │   │   └── tokens_per_message.png
│   │   │   ├── go.html
│   │   │   └── master.html
│   │   └── run.py
│   ├── data/
│   │   ├── DisasterResponse.db
│   │   ├── disaster_categories.csv
│   │   ├── disaster_messages.csv
│   │   └── process_data.py
│   ├── models/
│   │   ├── classifier.pkl
│   │   └── train_classifier.py
│   ├── .DS_Store
│   └── README.md
├── .gitattributes
├── .gitignore
├── DataBase.db
├── LICENSE
└── README.md
```

## Acknowledgements

All code, analysis and descriptions in this repository were made by this user and can be modyfied by anyone.


If you wantt to contribute to this project just fill a pull request.
