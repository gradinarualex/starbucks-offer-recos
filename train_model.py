# import libraries
import sys
import os
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


def load_data(data_path, target_col='Promotion'):
    ''' Function used to load the data from a file using pandas,
        to extract the features of interest (V1-V7) and target and
        returns features as X and target as y.
        
        Args:
            data_path (string): path to data file to read in the data
            target_col (optional): column name to be taken as target;
                default value is 'Promotion'
            
        Returns:
            X (2D array): array of features
            y (1D array): array of target values (0 or 1)
    '''
    
    # load data from file
    df = pd.read_csv(data_path)
    
    X = df.iloc[:, 3:] # get only feature columns
    y = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return X, y


def build_model():
    ''' Function to build a 2-class classification model.
        Build pipeline, define parameters, creates and returns
        Cross Validation model.
        
        Args:
            None
            
        Returns:
            cv (model): cross-validation classifier model
    '''
    
    pass


def evaluate_model(model, X_test, y_test):
    ''' Function to evaluate the performance of our model.
        Uses recall_score and f1_score to measure performance,
        as in this model it is most important to reduce false positives.
        
        Args:
            model:
            X_test:
            y_test:
            
        Returns:
            evaluation_scores (tuple): two-element tuple with first value
                as recall score and second as f1-score.
    '''
    
    pass


def save_model(model, model_filepath):
    ''' Function to save the fitted model to disk.
    
        Args:
            model: model to be saved on disk
            model_filepath: file path to where you want to save the model
            
        Returns:
            None
    '''
    
    filename = 'sb_recommender_model.sav'
    joblib.dump(model, filename)
    

def main():
    if len(sys.argv) == 3:
        data_path, model_path = sys.argv[1:]
        print('Loading data...\n    DATA FILE: {}'.format(data_path))
        X, y = load_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)
        
        print('Saving model...\n    MODEL: {}'.format(model_path))
        save_model(model, model_path)
        
        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the starbucks recommedations dataset '\
              'as the first argument and the filepath of the model pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/dataset.csv ./models/recommender_model.sav')
        

if __name__ == '__main__':
    main()