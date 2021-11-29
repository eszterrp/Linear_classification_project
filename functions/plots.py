#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:32:21 2021

@author: pazma
"""

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import roc_curve, roc_auc_score

def confusion_matrix_wo_crossval(X,y, y_corr, model, class_labels):
    
    
    #y_hat=model.predict(X)
    cm =  confusion_matrix(y_pred=y_corr, y_true=y, labels=class_labels)
    #print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()
    

def confusion_matrix_(X,y, y_corr, model, class_labels):
    
    
    #y_hat=model.predict(X)
    cm =  confusion_matrix(y_pred=y_corr, y_true=y, labels=class_labels)
    #print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()
    
# use in-sample cross-validation
    print("Confusion matrix of in-sample cross-validation:")
    
    
    y_hat_cv = cvp(model, X, y, cv=100)
    
    cm2 =  confusion_matrix(y_pred=y_hat_cv, y_true=y, labels=class_labels)
    #print(cm2)
    df_cm2 = pd.DataFrame(cm2, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm2, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()

    

# calculate the Euler number to the power of its coefficient to find the importance.
def feature_importance_plot(model):
    feature_importance = pd.DataFrame(feature_names, columns = ["feature"])  # feature names defines in the jupyter notebook
    feature_importance["importance"] = model.coef_[0]
    feature_importance["importance_abs_value"] = feature_importance["importance"].abs()

    feature_importance_top10 = feature_importance.sort_values(by = ["importance_abs_value"], ascending=True).head(15)

    fig = plt.figure(figsize = (20,25))
    ax = feature_importance_top10.plot.barh(x='feature', y='importance', 
                                               title="Top 15 most important variables according to their LogReg coefficients ")
    plt.show()



def get_auc(y, y_pred_probabilities, class_labels, column =1, plot = True):
    """Plots ROC AUC
    """
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities,drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities)
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