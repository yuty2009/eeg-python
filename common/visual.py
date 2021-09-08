# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_loss(train_loss, val_loss, save_path = None):
    '''Plot the training loss.'''
    plt.figure()
    plt.title("Training Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,len(train_loss)+1),train_loss,label="Train loss")
    plt.plot(range(1,len(val_loss)+1),val_loss,label="Validation Loss")
    plt.legend(loc='upper right')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_accu(train_accu, valid_accu, save_path= None):
    '''Plot the train and validation accuracy.'''
    plt.figure()
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,len(train_accu)+1),train_accu,label="Train Acc")
    plt.plot(range(1,len(valid_accu)+1),valid_accu,label="Validation Acc")
    plt.ylim((0,1.))
    plt.legend(loc='upper right')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(targets, predictions, target_names,
                          title='Confusion matrix', cmap="Blues",
                          save_path= None):
    """Plot Confusion Matrix."""
    cm = confusion_matrix(targets, predictions)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g = sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                    cmap=cmap)
    g.set_title(title)
    g.set_ylabel('True label')
    g.set_xlabel('Predicted label')
    return g