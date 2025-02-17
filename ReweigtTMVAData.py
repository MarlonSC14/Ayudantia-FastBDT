import os
os.system("wget http://root.cern/files/tmva_class_example.root")

import sys
sys.path.append("/root/scripts")

import ks_test
import uproot3 as uproot
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

cols = ['var1', 'var2', 'var3', 'var4']

def params_to_string(model):
  """Function to obtain all parameters of the model and save it to a string
  This string is later used to show the performance of the model by `plot_classifier_distributions`
  """
  model.get_params()
  string = ''
  lens = [len(k) for k in model.get_params()]
  max_len = max(lens)
  for k,v in model.get_params().items():
    #string += f'{k+" "*(max_len-len(k))} =  {v}\n'
    string += f'{k} = {v}\n'

  return string

def plot_classifier_distributions(model, test, train, print_params=False, round_numbers=4):
    """To evaluate the performance of the model, we first need to obtain the output distribution for test and train samples
    Then, evaluate the auc for each sample, and finnally the ks test of the distributions for each class
    print_params :
      Show all parameters of the model obtained from `model.get_params()`
    """

    test_background = model.predict_proba(test.query('label==0')[cols])[:,1]
    test_signal     = model.predict_proba(test.query('label==1')[cols])[:,1]
    train_background= model.predict_proba(train.query('label==0')[cols])[:,1]
    train_signal    = model.predict_proba(train.query('label==1')[cols])[:,1]

    test_pred = model.predict_proba(test[cols])[:,1]
    train_pred= model.predict_proba(train[cols])[:,1]

    density = True

    fig, ax = plt.subplots(figsize=(10, 7))

    background_color = 'red'

    opts = dict(
        range=[0,1],
        bins = 25,
        density = density
    )
    histtype1 = dict(
        histtype='stepfilled',
        linewidth=3,
        alpha=0.45,
    )

    ax.hist(train_background, **opts, **histtype1,
             facecolor=background_color,
             edgecolor=background_color,
             zorder=0)
    ax.hist(train_signal, **opts, **histtype1,
             facecolor='blue',
             edgecolor='blue',
             zorder=1000)






    hist_test_0 = np.histogram(test_background, **opts)
    hist_test_1 = np.histogram(test_signal, **opts)
    bins_mean = (hist_test_0[1][1:]+hist_test_0[1][:-1])/2
    bin_width = bins_mean[1]-bins_mean[0]
    area0 = bin_width*np.sum(test.label==0)
    area1 = bin_width*np.sum(test.label==1)

    opts2 = dict(
          capsize=3,
          ls='none',
          marker='o'
    )


    ax.errorbar(bins_mean, hist_test_0[0],  yerr = np.sqrt(hist_test_0[0]/area0), xerr=bin_width/2,
                 color=background_color, **opts2, zorder=100)
    ax.errorbar(bins_mean, hist_test_1[0],  yerr = np.sqrt(hist_test_1[0]/area1), xerr=bin_width/2,
                 color='blue', **opts2, zorder=10000)




    _ks_back = ks_test.ks_2samp_sci(train_background, test_background)[1]
    _ks_sign = ks_test.ks_2samp_sci(train_signal, test_signal)[1]

    print('Own ks test\n',
          ks_test.ks_2samp_weighted(train_background, test_background)[1],
          ks_test.ks_2samp_weighted(train_signal, test_signal)[1], sep='\n\t')

    auc_test  = roc_auc_score(test.label,test_pred )
    auc_train = roc_auc_score(train.label,train_pred)
    legend_elements = [Patch(facecolor='black', edgecolor='black', alpha=0.4,
                             label=f'Train (auc) : {round(auc_train,round_numbers)}'),
                      Line2D([0], [0], marker='|', color='black',
                             label=f'Test (auc) : {round(auc_test,round_numbers)}',
                              markersize=25, linewidth=1),
                       Circle((0.5, 0.5), radius=2, color='red',
                              label=f'Background (ks-pval) : {round(_ks_back,round_numbers)}',),
                       Circle((0.5, 0.5), 0.01, color='blue',
                              label=f'Signal (ks-pval) : {round(_ks_sign,round_numbers)}',),
                       ]

    ax.legend(
              #title='KS test',
              handles=legend_elements,
              #bbox_to_anchor=(0., 1.02, 1., .102),
              loc='upper center',
              ncol=2,
              #mode="expand",
              #borderaxespad=0.,
              frameon=True,
              fontsize=15)

    if print_params:
      ax.text(1.02, 1.02, params_to_string(model),
        transform=ax.transAxes,
      fontsize=13, ha='left', va='top')

    ax.set_yscale('log')
    ax.set_xlabel('XGB output')
    #ax.set_ylim(0.01, 100)

    #plt.savefig(os.path.join(dir_, 'LR_overtrain.pdf'), bbox_inches='tight')
    return fig, ax

tmvaData = uproot.open('tmva_class_example.root')
#print(tmvaData.keys())
signalData = tmvaData['TreeS']
signalDF = signalData.arrays(outputtype=pd.DataFrame)
signalDF['label'] = 1

backgroundData = tmvaData['TreeB']
backgroundDF   = backgroundData.arrays(outputtype=pd.DataFrame)
backgroundDF['label'] = 0

for var in cols:
  h = plt.hist(signalDF[var], bins=40, density=True, label='Signal')
  plt.hist(backgroundDF[var], bins=h[1], density=True,
           alpha=0.7, label='Background')
  #plt.hist(backgroundDF[var], bins=h[1], density=True,
  #         color='tab:orange', weights=backgroundDF.weight,
  #        label='Background - weighted', histtype='step')
  plt.xlabel(var)
  plt.legend(frameon=True)
  plt.show()
