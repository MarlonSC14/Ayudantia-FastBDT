import pandas as pd #Importing pandas to work with dataframes
import uproot3 as uproot #Importing uproot3 to import the datasets 
import matplotlib.pyplot as plt #Importing matplotlib.pyplot to visualize the results with histograms
from PyFastBDT import FastBDT #Importing FastBDT to train a model with our data and get the Fast Weights
import json #Importing json to get some necessary files with this extension
import numpy as np #Importing numpy to make some computings
import ROOT #Importing ROOT to perform de Chi2 tests in the histograms
import mplhep # Necessary library to use the script histos_weighted
import histos_weighted # Script to simplify plotting and necessary to compute X^2 
import plot_tools_simple # Script to compute X^2
import scalefactors #Script to read the files that contain the ScaleFactors
# note: The scripts histos_weighted, plot_tools_simple and scalefactors are python local files that you might not have

#Creating function to read the data more easily
def read_data(path):
    if path.endswith('csv'):
        return pd.read_csv(path)
    elif path.endswith('root'):
        root_file = uproot.open(path)['ntuple']
        pd_file = root_file.arrays(outputtype=pd.DataFrame)
        return pd_file
    elif path.endswith('json'):
        with open(path, 'r') as jsonfile:
            json_contain= json.load(jsonfile)
        return json_contain
    else:
        raise NotImplementedError
        print(f'Not ready, plase implement how your data should be handled!: \n -- {path}')

# Fixing the necessary values to identify and create the files
file_label = 'B+_nores'
label_chi = 'nores'
load_model = True
first= False
model = 'Reweighting_B+_jpsi.fbdt'
rows_df = 'df_rows_B+.csv'
results = 'df_B+.csv'

#Importing datasets, variables to use and hyperparameters to train the model
file_rd = 'RD_BinComplete_5_Cuts17_Fraction100_HLTAll_P1.root'
file_mc = 'BSLL_BinComplete_5.root'
file_json = 'options_Nominal.json'
file_sf = None
sw = 'sW_Cuts17_GaussDCB'
SF = 'TotalSF'

#Reading the data and transforming it to dataframes or dictionaries
mc_df = read_data(file_mc)
rd_df = read_data(file_rd)
json_contain = read_data(file_json)

#Getting the sweights from the RD dataframe
sWeights = rd_df[sw]

# To plot an histogram of the SWeights
plt.hist(sWeights, bins=100)
plt.title(f'sWeights histogram {file_label}')
plt.savefig(f'histo_weights_{file_label}.png')
plt.show()

# Visualizing the sweights along with the BMass variable (used to create the sweights)
plt.scatter(rd_df.BMass, sWeights)
plt.xlabel('BMass')
plt.ylabel('sWeights')
plt.title(f'{file_label}')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'bmass_weights_scatter_{file_label}.png')
plt.show()

#We should only use data inside the region of the Bmass variable visualized in the last plot
mc_df = mc_df.query(f'{min(rd_df.BMass)}<=BMass<={max(rd_df.BMass)}')
rd_df = rd_df.query(f'{min(rd_df.BMass)}<=BMass<={max(rd_df.BMass)}')

#Geting scalefactors and Sweights into the dataframes
rd_df['weights'] = sWeights
if file_sf!=None:
    scale_factors = read_data(file_sf)
    scalefactors.apply_SF_event(mc_df, 'Soft', file=file_sf)
    mc_df["weights"] = mc_df['SF'+'Soft']
elif SF in mc_df:
    mc_df["weights"] = mc_df[SF]
else:
    mc_df["weights"] = 1

#Using only the features that are of our interest
cols = [var for var in json_contain['variables'] if var in rd_df.keys()]
ctrl_cols = [var for var in json_contain['control_variables'] if var in rd_df.keys()]
ctrl_mc_df = mc_df[ctrl_cols]
ctrl_rd_df = rd_df[ctrl_cols]
mc_df=mc_df[cols+['weights']]
rd_df=rd_df[cols+['weights']]
mc_df['label']=0
rd_df['label']=1

#Creating and training a FastBDT model or Loading a pre existent model
if load_model: 
    classifier = FastBDT.Classifier()
    classifier.load(model)
else:
    #Creating a complete df (using montecarlo data and real data) dataframe to train our model 
    cols_train_rd = cols + ['label', 'weights']
    cols_train_mc = cols + ['label', 'weights']
    dataFrame = pd.concat([rd_df[cols_train_rd],
                           mc_df[cols_train_mc]])

    param_dict = json_contain['model_params']
    classifier = FastBDT.Classifier(**param_dict)
    classifier.fit(X=dataFrame[cols], 
                    y=dataFrame.label, 
                    weights=dataFrame['weights']
                    )
    classifier.save(f'Reweighting_{file_label}.fbdt')

#Geting the probabilities of each event to be signal and calculating weights
predict = classifier.predict(mc_df[cols])
fastW = predict/(1-predict)

#Processing and correcting some possible errors in the computing of the Fast Weights
weight_mean, weight_std = np.mean(fastW), np.std(fastW)
large_fastW = np.abs(fastW-weight_mean)>10*weight_std
if np.sum(large_fastW)>0:
        print('WARNING! EXTRA LARGE fastW! SETTING THEM TO ZERO!')
        print(f'Number of fastW 10 sigma far from the mean: {np.sum(large_fastW)}')
        fastW = np.where(large_fastW, 0, fastW)

fastW *= len(mc_df)/np.sum(fastW)
weight_var = f'fastW'
mc_df[weight_var] = fastW        
mc_df['TotalWeight'] = mc_df[weight_var]*mc_df['weights']

#Saving the weights computed in a .txt file
with open(f'weights_{file_label}.txt', 'w') as f:
       for w in fastW:
              f.write(str(w) + '\n')

#Recovering the control variable columns in the main dataframe
rd_df[ctrl_cols]=ctrl_rd_df[ctrl_cols]
mc_df[ctrl_cols]=ctrl_mc_df[ctrl_cols]

#Renaming some variables to simplify the process
target_data=rd_df
rew_data=mc_df

#Creating helper variables
bins=100
X2_or = []
X2_rew = []
cols = cols + ctrl_cols

#Generating plots to visualize the results and calculate the Chi^2/DOF metric
for var in cols:   
        fig = plt.figure(figsize=[14,12])
        axes = plot_tools_simple.create_axes_for_pulls(fig)
    
        real = target_data[var]
        mc   = rew_data[var]

        on_p = np.percentile(real, 1)
        ni_p = np.percentile(real, 99)

        histo_raw = plot_tools_simple.hist_weighted(mc, 
                                    #range=[on_p, ni_p], 
                                    bins=bins, marker='o', axis=axes[0], density=True, color='orangered', label='MC')


        histo_final = plot_tools_simple.hist_weighted(mc, weights=rew_data['TotalWeight'], 
                                            bins=histo_raw[1],
                                            marker='o', axis=axes[0], density=True, color='seagreen', label='Reweighted MC')


        histo_data = plot_tools_simple.hist_weighted(real, weights=target_data['weights'], 
                                            bins=histo_raw[1], 
                                            marker='s', axis=axes[0], density=True, color='blue', 
                                            label='Data sWeight', markersize=7.5)


        bin_mean = (histo_raw[1][1:]+histo_raw[1][:-1])/2
        bin_err = (histo_raw[1][1:]-histo_raw[1][:-1])/2
        axes[0].set_ylim(0)
        #axes[0].set_xlim(on_p, ni_p)
        axes[0].set_ylabel(f'Density / {round(bin_err[0]*2, 4)}')
        axes[0].legend(frameon=True, fontsize=15)

        ls = 'none'
        ratio_raw = plot_tools_simple.create_ratio(histo_raw, histo_data)
        ratio_raw = list(ratio_raw)
        mean_errrs = np.mean(ratio_raw[1])
        std_errrs  = np.std(ratio_raw[1])
        remove_big_errors = ratio_raw[1]<mean_errrs+10*std_errrs        
        ratio_raw[0] = ratio_raw[0][remove_big_errors]
        ratio_raw[1] = ratio_raw[1][remove_big_errors]

        chi2_raw = plot_tools_simple.get_chi2(histo_raw, histo_data)
        label_chi2 = str(round(chi2_raw, 1))
        label_chi2 = str(round(chi2_raw/(len(histo_data[0])-1), 1))
        label_chi2_raw = label_chi2
        axes[1].errorbar(bin_mean[remove_big_errors], ratio_raw[0], ratio_raw[1], bin_err[remove_big_errors], color='orangered', marker='o', ls=ls, capsize=2, label=label_chi2)

        ratio_final = plot_tools_simple.create_ratio(histo_final, histo_data)
        ratio_final = list(ratio_final)
        mean_errrs = np.mean(ratio_final[1])
        std_errrs  = np.std(ratio_final[1])
        #remove_big_errors = ratio_final[1]<mean_errrs+10*std_errrs
        ratio_final[0] = ratio_final[0]
        ratio_final[1] = ratio_final[1]
        chi2_final = plot_tools_simple.get_chi2(histo_final, histo_data)
        label_chi2 = str(round(chi2_final, 1))
        label_chi2 = str(round(chi2_final/(len(histo_data[0])-1), 1))
        label_chi2_final = label_chi2
        axes[1].errorbar(bin_mean, ratio_final[0], ratio_final[1],  color='seagreen', marker='o', ls=ls, capsize=2, label=label_chi2,zorder=200)
        axes[1].axhline(1, ls='--', color='blue')
        axes[1].set_xlabel(var)
        axes[0].set_xlabel('')
        axes[1].set_ylabel('Ratio w.r.t. Data', loc='center')
        axes[1].legend(frameon=True, title=r'$\chi^2/nBins$   w.r.t. Data', ncol=3, fontsize=14, title_fontsize=14)
        axes[1].axhline(y=1, color='r', linestyle='-')
        plot_tools_simple.hep.cms.label(ax=axes[0], label='Preliminary', data=True, com=13.6, year=2022)
        plt.savefig(f'histo_pulls_{file_label}_{var}.png')
        print(mean_errrs)
        if mean_errrs<1:        
            axes[1].set_ylim(0.5, 1.5)
        else:
            axes[1].set_ylim(0, 2)
        plt.show()

        histo_rd_sw = plt.hist(real, bins=bins, density=True, label='Data', weights=target_data['weights'])
        histo_sf = plt.hist(mc, bins=histo_rd_sw[1], density=True, histtype='step', linewidth=2,label='MC')#, weights=mc_df['weights'])
        histo_w = plt.hist(mc, bins=histo_rd_sw[1], density=True, histtype='step', linewidth=2, label='MC reweighted', weights=rew_data['TotalWeight'])
        plt.title(f'{var}')
        plt.legend(frameon=True)
        for i in range(len(histo_rd_sw[0])):
            if histo_rd_sw[0][i]>=0.01*max(histo_rd_sw[0]):
                min_lim=max(rd_df[var])*i/bins
                bin=i
                break
        if bin>5:
            max_lim = max(rd_df[var])
        else:
            for j in range(bin,len(histo_rd_sw[0])):
                if histo_rd_sw[0][j]<=0.01*max(histo_rd_sw[0]):
                    max_lim = max(rd_df[var])*j/bins
                    break
        plt.xlim(min_lim,max_lim)
        plt.text(x=min(histo_rd_sw[0]),y=0,s=f'X^2_or/DOF={label_chi2_raw} \nX^2/DOF={label_chi2_final}')
        plt.legend(frameon=True)
        plt.savefig(f'histo_{file_label}_{var}.png')
        plt.show()
        X2_or.append(label_chi2_raw)
        X2_rew.append(label_chi2_final)

chi_results_data = {f'X2_{label_chi}_or': X2_or,
                    f'X2_{label_chi}_rew': X2_rew}
chi_results = pd.DataFrame(chi_results_data)
if first:
    data = {'vars': cols}
    results_df = pd.DataFrame(data)
    results_df.to_csv(rows_df, index=False)
else: 
    results_df = pd.read_csv(results)
results_df[f'X2_{label_chi}_or'] = chi_results_data[f'X2_{label_chi}_or']
results_df[f'X2_{label_chi}_rew'] = chi_results_data[f'X2_{label_chi}_rew']
results_df.to_csv(results, index=False)