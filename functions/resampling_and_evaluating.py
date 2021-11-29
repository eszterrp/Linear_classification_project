#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:30:56 2021

@author: pazma
"""

from numpy import mean
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek 


def sampling_and_evaluate(X,y,sampling, sampling_strat): 
    RANDOM_STATE = 42
    
    
    
    if sampling =="under":  
        
         # define undersampling strategy
        undersample = RandomUnderSampler(sampling_strategy=sampling_strat, random_state=RANDOM_STATE)
        X_under, y_under = undersample.fit_resample(X, y)
        
        # summarize class distribution
        #print(Counter(y_under))
        lregr = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500, random_state=RANDOM_STATE)

        # fit model
        lregr.fit(X_under, y_under)
        
        # make prediction
        y_hat_under=lregr.predict_proba(X_under)

        
        # evaluate model
        return roc_auc_score(y_true=y_under, y_score=y_hat_under[:,1])
        #print('AUC Score of UnderSampling with', str(sampling_strat), 
        #      'sampling strategy: ', roc_auc_score(y_true=y_under, y_score=y_hat_under[:,1]))
        
    elif sampling =="tomek": 
        
        # define undersampling strategy
        from imblearn.under_sampling import TomekLinks

        tomek = TomekLinks( sampling_strategy=sampling_strat) #random_state= 42
        # unexpected error: TypeError: __init__() got an unexpected keyword argument 'random_state
        # so that is why the random_state argument is not used in this case
        
        X_t, y_t = tomek.fit_resample(X, y)
        
        # summarize class distribution
        #print(Counter(y_t))
        lregr = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500, random_state=RANDOM_STATE)

        # fit model
        lregr.fit(X_t, y_t)
        
        # make prediction
        y_hat_tomek=lregr.predict_proba(X_t)

        
        # evaluate model
        return roc_auc_score(y_true=y_t, y_score=y_hat_tomek[:,1])
        #print('AUC Score of UnderSampling using Tomek-links with', str(sampling_strat), 
         #     'sampling strategy: ', roc_auc_score(y_true=y_t, y_score=y_hat_tomek[:,1]))
        
    elif sampling=="over":
        
        # define oversampling strategy
        oversample = RandomOverSampler(sampling_strategy=sampling_strat, random_state=RANDOM_STATE)
        X_over, y_over = oversample.fit_resample(X, y)
        
        # summarize class distribution
        #print(Counter(y_over))
        lregr = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500, random_state=RANDOM_STATE)

        # fit model
        lregr.fit(X_over, y_over)
        
        # make prediction
        y_hat_over=lregr.predict_proba(X_over)

        
        # evaluate model        
        return roc_auc_score(y_true=y_over, y_score=y_hat_over[:,1])
        #print('AUC Score of OverSampling with', str(sampling_strat), 
        #      'sampling strategy: ', roc_auc_score(y_true=y_over, y_score=y_hat_over[:,1]))
        
    elif sampling=="smote":
        
        # define oversampling strategy    
        smote = SMOTE(sampling_strategy=sampling_strat, random_state=RANDOM_STATE)
        X_sm, y_sm = smote.fit_resample(X, y)
        
        # summarize class distribution
        #print(Counter(y_sm))
        lregr = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500, random_state=RANDOM_STATE)

        # fit model
        lregr.fit(X_sm, y_sm)
        
        # make prediction
        y_hat_sm=lregr.predict_proba(X_sm)

        
        # evaluate model
        return roc_auc_score(y_true=y_sm, y_score=y_hat_sm[:,1])
        #print('AUC Score of OverSampling using SMOTE with', str(sampling_strat), 
        #      'sampling strategy: ', roc_auc_score(y_true=y_sm, y_score=y_hat_sm[:,1]))
    elif sampling=="smotetomek": 
        
        
        
        smotetomek = SMOTETomek(sampling_strategy=sampling_strat, random_state=RANDOM_STATE)
        X_smtl, y_smtl = smotetomek.fit_resample(X, y)

        # summarize class distribution
        #print(Counter(y_sm))
        lregr = LogisticRegression(penalty='l2', C=100.0, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           solver='liblinear', max_iter=500, random_state=RANDOM_STATE)

        # fit model
        lregr.fit(X_smtl, y_smtl)

        # make prediction
        y_hat_smtl=lregr.predict_proba(X_smtl)


        # evaluate model
        return roc_auc_score(y_true=y_smtl, y_score=y_hat_smtl[:,1])



def test_evaluate_sampling_ratios(ratios,ratios_reversed, result_dictionary ):

    for r1 in ratios:
        for r2 in ratios_reversed: 
            try:

                # define oversampling strategy
                over = RandomOverSampler(sampling_strategy=r1)
                # fit and apply the transform
                X_over, y_over = over.fit_resample(X_scaled, y_original)

                # define undersampling strategy
                under = RandomUnderSampler(sampling_strategy=r2)
                # fit and apply the transform
                X_over_under, y_over_under = under.fit_resample(X_over, y_over)

                # use a Logistic Regression setting based on previous finetuning results
                # (high C value)
                lregr = LogisticRegression(penalty='l2', C=1000.0, 
                                           fit_intercept=True, 
                                           intercept_scaling=1, 
                                           solver='liblinear', max_iter=500)

                # fit model
                lregr.fit(X_over_under, y_over_under)

                # make prediction
                y_hat_overunder=lregr.predict_proba(X_over_under)


                # evaluate model
                dic_key=str("OverSampler ratio: " + str(r1) + ", UnderSampler ratio: " + str(r2))
                result_dictionary[dic_key]=roc_auc_score(y_true=y_over_under, y_score=y_hat_overunder[:,1])

                # store ratios of best sampling strategy with the highest AUC score
                sorted(result_dictionary.items(), key=lambda item: item[1], reverse=True)[:1]
                l=list(dict(sorted(result_dictionary.items(), key=lambda item: item[1], reverse=True)[:1]).keys())
                ratios=list(flatten([re.findall(r"0\.\d{1}", x) for x in l]))
                overs_r=float(ratios[0])
                unders_r=float(ratios[1])
                return overs_r, unders_r
            except ValueError: 
                pass



def evaluate_best_sampling(overs_r, unders_r, model):
    
    
    
    over = RandomOverSampler(sampling_strategy=float(overs_r))
    # fit and apply the transform
    X_over, y_over = over.fit_resample(X_scaled, y_original)

    # define undersampling strategy
    under = RandomUnderSampler(sampling_strategy=float(unders_r))
    # fit and apply the transform
    X_over_under, y_over_under = under.fit_resample(X_over, y_over)

    model = LogisticRegression(penalty='l2', C=1000.0,  
                               solver='liblinear', max_iter=500)


    model.fit(X_over_under, y_over_under)
    y_hat=model.predict_proba(X_over_under)

    from sklearn import metrics
    get_auc(y_over_under,y_hat[:,1] , class_labels, column=1, plot=True) 

    # Classification report
    print("Classification report of the model: \n", 
          metrics.classification_report(y_over_under, model.predict(X_over_under)))