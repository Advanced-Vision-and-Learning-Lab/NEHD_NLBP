# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:45:03 2020
Function to generate loss and accuracy curves
@author: jpeeples
"""

import matplotlib.pyplot as plt
import numpy as np
import torch ##
import pdb

def Plot_Learning_Curves(train_acc,train_loss,val_acc,val_loss,best_epoch,
                         sub_dir,weight=None):
    
    # Turn interactive plotting off, don't show plots
    plt.ioff()
    
    #For each type of loss in dictionary, plot training and validation loss
    # in each subplots
    
    loss_fig = plt.figure()
    
    count = 0
        
    loss_ax = loss_fig.add_subplot(1, 1, 1)
    epochs = np.arange(1,len(train_loss)+1)
    loss_ax.plot(epochs,train_loss)
    loss_ax.plot(epochs,val_loss)

    ymin = val_loss[best_epoch]
    text= "Epoch={}, Val_loss={:.3f}".format(best_epoch+1, ymin)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    loss_ax.annotate(text, xy=(best_epoch+1, ymin), xytext=(0.94,0.96), **kw)

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Error')
    count += 1
    
    loss_fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
    loss_fig.subplots_adjust(right=0.75)
    loss_ax.legend(['Training', 'Validation'],
              bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    
    plt.suptitle('Loss Curves for {} Epochs'.format(len(train_loss)))
    loss_fig.savefig((sub_dir + 'Loss Curves.png'), dpi=loss_fig.dpi)
    plt.close(loss_fig)
        
    # visualize results
    acc_fig, acc_ax = plt.subplots()
    val_acc_tensor = torch.tensor(val_acc)
    val_acc_np = val_acc_tensor.cpu().numpy()

    train_acc_tensor = torch.tensor(train_acc)
    train_acc_np = train_acc_tensor.cpu().numpy()

    acc_ax.plot(epochs,train_acc_np)
    acc_ax.plot(epochs, val_acc_np)
    ymax = val_acc[best_epoch]
    text= "Epoch={}, Val_acc={:.3f}".format(best_epoch+1, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    acc_ax.annotate(text, xy=(best_epoch+1, ymax), xytext=(0.94,0.96), **kw)
    plt.suptitle('Accuracy Curves for {} Epochs'.format(len(train_acc)))
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='best')
    acc_fig.savefig((sub_dir + 'Accuracy Curve.png'), dpi=acc_fig.dpi)
    plt.close(acc_fig)
