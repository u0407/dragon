from hmmlearn.hmm import GMMHMM
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
import scipy.stats as scs 
import glob 
import os 
from dotenv import load_dotenv
load_dotenv()

with open('./part.txt', 'r', encoding='utf-8') as file:
    part = file.read()
project_pth = f'c:/Users/shen_/Code/dragon/GA_Shane/outputs/{part}'

os.chdir(project_pth)

file = glob.glob(project_pth+'/*_output_axis.csv')[0]

os.makedirs(project_pth+'/picture',exist_ok=True)
os.chdir(project_pth+'/picture')

print("Project: ",part)
print("file: ",file)

for L in [5,7,9]:
    
    diff = 1     ### 1是有HLdiff，0是没有HLdiff
    mix = 3    ### GMM mix参数

    df = pd.read_csv(file)
    close = df['close']
    high = df['high'][L:]
    low = df['low'][L:]
    eob = df['eob']
    datelist = pd.to_datetime(eob[L:])

    logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[(L-1):]
    logreturnX = np.log(np.array(close[L:]))-np.log(np.array(close[:-L]))
    HLdiff = (np.log(np.array(high))-np.log(np.array(low)))

    closeidx = close[L:]

    if diff == 1:
        X = np.column_stack([HLdiff ,logreturnX,logreturn])
        
    else:
        X = np.column_stack([logreturnX,logreturn])

    gmm = GMMHMM(n_components = 2, n_mix=mix, covariance_type='diag', n_iter = 369, random_state = 369).fit(X)

    latent_states_sequence = gmm.predict(X)
    latent_states_proba = gmm.predict_proba(X)[:,0]
    latent_states_proba = np.abs(latent_states_proba-0.5)
    
    sns.set_style('white')
    plt.figure(figsize = (20, 8))
    for i in range(gmm.n_components):
        state = (latent_states_sequence == i)
        plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
        plt.legend()
        plt.grid(1)

    plt.savefig(f"Label_{L}.jpg")     


    data = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')
    
    plt.figure(figsize=(20,8))
    for i in range(gmm.n_components):
        state = (latent_states_sequence == i)
        idx = np.append(0,state[1:])
        data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 
        plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'latent_state %d'%i)
        plt.legend(loc='upper left')
        plt.grid(1)
        plt.title('label wo shifting')

    plt.savefig(f"Label_a_{L}.jpg")     

    data = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')

    plt.figure(figsize=(20,8))
    for i in range(gmm.n_components):
        state = (latent_states_sequence == i)
        idx = np.append(0,state[:-1])
        data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 
        plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'latent_state %d'%i)
        plt.legend(loc='upper left')
        plt.grid(1)
        plt.title('label shifting')
        
    plt.savefig(f"Label_b_{L}.jpg")     ##############  保存图片    #####################


    # 自动判断，将标签的 1与0，变为buy为1，sell为0
    data = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')
    for i in range(gmm.n_components):
        state = (latent_states_sequence == i)
        idx = np.append(0,state[:-1])
        data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 

    if sum(data['state 0_return']) > sum(data['state 1_return']):
        data['state'] = abs(data['state'] - 1)
        
    # 标签数据导出
    data.to_csv(file[:-4] + "_Label_" + str(L) + str(diff) + str(mix) + ".csv")   
