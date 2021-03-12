# import libraries
import sys
import os
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

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
    
    X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']].values # get only feature columns
    y = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0).values # get only target column
    
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


def evaluate_model(model, X_test, y_test, verbose=True):
    ''' Function to evaluate the performance of our model.
        Uses precision, recall and f1 scores to measure performance,
        as in this model it is most important to minimize false positives
        and false negatives.
        
        Args:
            model: model to be evaluated
            X_test: test feature matrix
            y_test: true values of target array
            
        Returns:
            evaluation_scores (dict): three-element dictionsr with
            precision score, recall score and third as f1-score.
    '''
    
    # get model predictions
    y_pred = model.predict(X_test)
    
    # create score dataframe
    score_df = pd.DataFrame({
        'offered': y_pred,
        'purchased': y_test
    })
    
    cust_treat = (score_df['offered'] == 1).sum() # number of customers offered (treatment)
    cust_cntrl = (score_df['offered'] == 0).sum() # number of customers not offered (control)

    purch_treat = ((score_df['offered'] == 1) & (score_df['purchased'] == 1)).sum() # number of offers used (true positives)
    purch_cntrl = ((score_df['offered'] == 0) & (score_df['purchased'] == 1)).sum() # number of customer not offered that purchased (false negatives)
    
    # incremental response rate calculation
    irr = purch_treat / cust_treat - purch_cntrl / cust_cntrl
    # net incremental revenue
    nir = (10 * purch_treat - 0.15 * cust_treat) - 10 * purch_cntrl
    
    # create scores dictionary
    evaluation_scores = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'irr': irr,
        'nir': nir
    }
    
    # if verbose print scores
    if verbose:
        print('   Precision Score:                  {}'.format(evaluation_scores['precision']))
        print('   Recall Score:                     {}'.format(evaluation_scores['recall']))
        print('   F1 Score:                         {}'.format(evaluation_scores['f1']))
        print('   Incremental Response Rate (IRR):  {}'.format(evaluation_scores['irr']))
        print('   Net Incremental Revenue (NIR):    {}'.format(evaluation_scores['nir']))
    
    return evaluation_scores


def save_model(model, model_filepath):
    ''' Function to save the fitted model to disk.
    
        Args:
            model: model to be saved on disk
            model_filepath: path to where you want to save the model
            
        Returns:
            None
    '''
    
    # save model to disk
    joblib.dump(model, model_filepath)
    

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
              'train_classifier.py ../data/dataset.csv ./models/sb_recommender_model.sav')
        

if __name__ == '__main__':
    main()