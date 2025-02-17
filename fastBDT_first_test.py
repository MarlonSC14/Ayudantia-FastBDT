import pandas as pd #Importing pandas to work with dataframes
import uproot3 as uproot #Importing uproot3 to import the datasets 
import matplotlib.pyplot as plt #Importing matplotlib.pyplot to visualize the results with histograms
from PyFastBDT import FastBDT #Importing FastBDT to train a model with our data and get the weights
import json #Importing json to get the features to train with
import numpy as np #Importing numpy to make some computings
import ROOT #Importing ROOT to perform de Chi2 tests in the histograms
import mplhep
import histos_weighted
import plot_tools_simple

def get_chi2_mod(histo_num, histo_den):
    diff = histo_num[0]-histo_den[0]
    err = np.hypot(histo_num[-1], histo_den[-1])
    pull = diff/err
    isnan = np.isnan(pull)
    return np.sum(np.power(pull[np.logical_not(isnan)],2))

#Importing datasets, variables to use and hyperparameters to train the model
mc_file=uproot.open('JPSI_Bin3_5.root')['ntuple']
rd_file=uproot.open('RD_Bin3_5_Cuts17_Fraction10_HLTAll_P1.root')['ntuple']
with open('options_Nominal.json', 'r') as jsonfile:
        json_contain = json.load(jsonfile)

#Geting pandas dataframes
mc_df = mc_file.arrays(outputtype=pd.DataFrame)
rd_df = rd_file.arrays(outputtype=pd.DataFrame)

# To plot an histogram of the SWeights
# plt.hist(rd_df.sW_Cuts17_GaussDCB, bins=100)
# plt.show()

# The zeros in the sweights come from data not used to produce the sPlots
# plt.scatter(rd_df.BMass, rd_df.sW_Cuts17_GaussDCB)
# plt.xlabel('BMass')
# plt.ylabel('sWeights')
# plt.show()

#So we should only use data with BMass in [5.0, 5.7]
mc_df = mc_df.query('5<=BMass<=5.7')
rd_df = rd_df.query('5<=BMass<=5.7')

#Geting scalefactors and Sweights into the dataframes
SF=mc_df['TotalSF']
rd_df['sweights']=rd_df['sW_Cuts17_GaussDCB']
mc_df['sweights']=mc_df['TotalSF']

#Using only the features that are of our interest
cols = json_contain['variables']
mc_df=mc_df[cols+['sweights']]
rd_df=rd_df[cols+['sweights']]
mc_df['label']=0
rd_df['label']=1

#Creating a complete (using montecarlo data and real data) dataframe to train our model 
cols_train_rd = cols + ['label', 'sweights']
cols_train_mc = cols + ['label', 'sweights']
dataFrame = pd.concat([rd_df[cols_train_rd],
                           mc_df[cols_train_mc]])

#Training a FastBDT model
param_dict = json_contain['model_params']
classifier = FastBDT.Classifier(**param_dict)
classifier.fit(X=dataFrame[cols], 
                    y=dataFrame.label, 
                    weights=dataFrame['sweights']
                    )

#Geting the probabilities of each event to be signal and calculating weights
predict = classifier.predict(mc_df[cols])
fastW = (1-predict)/predict

# weight_mean, weight_std = np.mean(fastW), np.std(fastW)
# large_fastW = np.abs(fastW-weight_mean)>10*weight_std
# if np.sum(large_fastW)>0:
#         print('WARNING! EXTRA LARGE fastW! SETTING THEM TO ZERO!')
#         print(f'Number of fastW 10 sigma far from the mean: {np.sum(large_fastW)}')
#         fastW = np.where(large_fastW, 0, fastW)


#Ploting histograms of the real and motecarlo data along with a weighted histogram for each feature
for variable in cols:
        #n_mc, bins_mc = plt.hist(mc_df[variable], bins=100, density=True, label='MC')
        histo_rd_sw = plt.hist(rd_df[variable], bins=100, density=True, label='Data', weights=rd_df['sweights'])
        histo_sf = plt.hist(mc_df[variable], bins=100, density=True, histtype='step', linewidth=2,label='MC ScaleFactor', weights=SF)
        histo_w = plt.hist(mc_df[variable], bins=100, density=True, histtype='step', linewidth=2, label='MC reweighted', weights=SF+fastW)
        plt.title(f'{variable}')
        plt.legend(frameon=True)
        plt.savefig(f'hsto_{variable}.png')
        n_mc_sf_root = len(histo_sf[0])-1
        n_rd_sw_root = len(histo_rd_sw[0])-1
        n_mc_w_root = len(histo_w[0])-1
        root_h_mc_sf = ROOT.TH1F(f"root_h_mc_sf_{variable}", f"ROOT_1_{variable}", n_mc_sf_root, histo_sf[1][0], histo_sf[1][-1])
        root_h_rd_sw = ROOT.TH1F(f"root_h_rd_sw_{variable}", f"ROOT_2_{variable}", n_rd_sw_root, histo_rd_sw[1][0], histo_rd_sw[1][-1])
        root_h_mc_w = ROOT.TH1F(f"root_h_mc_w_{variable}", f"ROOT_3_{variable}", n_mc_w_root, histo_w[1][0], histo_w[1][-1])
        for i in range(n_mc_sf_root):
                root_h_mc_sf.SetBinContent(i+1, histo_sf[0][i])
        for i in range(n_rd_sw_root):
                root_h_rd_sw.SetBinContent(i+1, histo_rd_sw[0][i])
        for i in range(n_mc_w_root):
                root_h_mc_w.SetBinContent(i+1, histo_w[1][i])
        chi2_or = root_h_mc_sf.Chi2Test(root_h_rd_sw, option="UU")
        chi2_w = root_h_rd_sw.Chi2Test(root_h_mc_w, option="UW")
        print("ROOT chi2:")
        print(f'Original {variable}: {chi2_or}')
        print(f'Reweighted {variable}: {chi2_w}')
        # chi2_raw = get_chi2_mod(histo_sf,histo_rd_sw)
        # chi2_final = get_chi2_mod(histo_w,histo_rd_sw)
        # print("Horacio's implementation:")
        # print(f'Original {variable}: {chi2_raw}')
        # print(f'Reweighted {variable}: {chi2_final}')
        # plt.text(x=plt.xlim()[0]+0.05,y=plt.ylim()[1]-0.05, s=f'Raw X2={chi2_or} \n Reweight X2={chi2_w}')
        plt.show()