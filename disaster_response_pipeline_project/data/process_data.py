import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """

    Loads data from .csv file on filepath provided and merge both messages
    and categories dataframes


    Parameters:
        messages_filepath: Path pointing to the messages dataset
        categories_filepath: Path pointing to the categories dataset

    Returns:
        df: merged DataFrame
    """


    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')

    return df



def clean_data(df):
    """
    Cleans data, remove duplicates, creates new columns and change data types

    Parameters:
        df: DataFrame to be cleaned


    Returns:
        df: cleaned DataFrame

    """

    ## df has am inefficient catogorization
    ## So, the 'categories' column is used to get the columns names
   
    categories = df.categories.str.split(pat=';',expand=True)
   
    ## to do so, only the data from one row is manipulated to generate 
    ## columns names
    row = categories.loc[0]

    ## strips only the last 2 characters
    category_colnames = row.str[:-2]
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')




    df = df.drop(labels='categories',axis=1)

    df = pd.concat([df,categories],axis=1)

    df = df[df["related"] <= 1]

    df = df.drop_duplicates()

    return df




def save_data(df, database_filename):

    """
    Takes a clean dataframe with concatenated columns and save it into a
    table inside of a SQL database stored on database_filename provided

    Parameters:
        df: The DataFrame to persist
        database_filename: The database name

    """
    
    engine = create_engine('sqlite:///' + database_filename) #nome do arquivo
    df.to_sql("merged_df", engine, index=False, if_exists='replace') #nome da tabela


def main():

    """
    Main function of the program

    Parse the arguments from terminal and returns the currently step
    """


    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('\n Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()


# estando em Disaster-Response-Pipeline
# python disaster_response_pipeline_project//data//process_data.py disaster_response_pipeline_project//data//disaster_messages.csv disaster_response_pipeline_project//data//disaster_categories.csv merged_df.db

# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db