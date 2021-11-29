#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:31:58 2021

@author: pazma
"""
import os
os.chdir("/Users/pazma/Documents/BSE/cml/project-2_linear_classification/functions")

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import reweighting_predicted_results


def predict_and_evaluate_binary(X,y, model, crossval="Yes"):
    
    
    
    model.fit(X,y)
    y_hat=model.predict_proba(X)
    
    # reweighting 
    q1 = y.sum()/len(y)
    r1 = 0.5
    y_hat_corr=reweighting_predicted_results.reweight(y_hat[:,1], q1,r1)
    
    
    ### Evaluate Model ###
    y_pred_new = [1 if pi >= 0.25 else 0 for pi in y_hat_corr]
    
    # Confusion Matrix
    print("Confusion Matrix \n")
    # insample_labels = model.predict(X)
    cm =  confusion_matrix(y_pred=y_pred_new, y_true=y, labels=[0,1])
    print (cm)
    
    # Plotting confusion matrix (custom help function)
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],  # class labels are defined in the notebook
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()
    
    # ROC AUC score
    #get_auc(y_original,y_hat_corr , class_labels, column=1, plot=True) 
    fpr, tpr, _ = roc_curve(y == 1, y_hat_corr,drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_hat_corr)
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    

    # Classification report
    print("Classification report of the model: \n", 
      metrics.classification_report(y,y_pred_new))
    
    if crossval=="Yes":
        # cross validation
        print("Confusion matrix of in-sample cross-validation:")

        from sklearn.model_selection import cross_val_predict as cvp
        y_hat_cv = cvp(model, X, y, cv=100)

        cm2 =  confusion_matrix(y_pred=y_hat_cv, y_true=y, labels=[0,1])
        #print(cm2)
        df_cm2 = pd.DataFrame(cm2, index = [i for i in class_labels],
                      columns = [i for i in class_labels])
        sns.set(font_scale=1)
        sns.heatmap(df_cm2, annot=True, fmt='g', cmap='Blues')
        plt.xlabel("Predicted label")
        plt.ylabel("Real label")
        plt.show()
        
    elif crossval=="No":
        print("No cross-validation")


def predict_evaluate_multiclass(X,y,model):
    
    model.fit(X, y)

    # define the evaluation procedure with cross-validation accuracy scores
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate the model
    score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # mean accuracy 
    print("Mean accuracy result of in-sample cross-validation:", np.mean(score))
    
    y_hat=model.predict(X)
    class_report=metrics.classification_report(y,y_hat )
    print("\n")
    print(class_report)