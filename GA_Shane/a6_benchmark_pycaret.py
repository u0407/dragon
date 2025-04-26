# %%
import numpy as np
import pandas as pd
from pycaret.classification import *
import datetime
import matplotlib.pyplot as plt
import glob 
import os 

test_start = '2023-01-01'
fix = 713
data_1_size = 2983     ###### 测试数据行数  ###############

part = '20250426_023509_YvmTj7'
dir = f'/home/dragon/GA_Shane/outputs/{part}/'
os.chdir(dir)
os.makedirs('./temp',exist_ok=True)
# Remove all files that start with numbers in the current directory
for file in os.listdir('./temp'):
    if file[0].isdigit():
        os.remove('./temp/'+file)

train_path = glob.glob(os.path.join(dir,f'*_Train_{fix}_*.csv'))[0]
test_path = glob.glob(os.path.join(dir,f'*_Test_{fix}_*_PCA.csv'))[0]

assert str(data_1_size) in test_path

dataset_s = pd.read_csv(train_path)   ############# 训练集文件 ####################
dataset = dataset_s

num_xunlian = len(dataset_s)

# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)   ####替换正负inf为NA


m_size = 25     ####### 测试多少个月 #######
buy = 1     ##### 多 ###################
sell = 0     ##### 空 ####################
rrr = 0.25     ###### 系数 ###################
m = 1000     ###### 总资金 ###################

print(train_path)
print(test_path)

assert len(dataset['A0'].unique()) > 1

dataset.tail()

# %%
dataset = dataset.drop(columns=['A0_p','eob'])

# %%
from sklearn.decomposition import PCA

num = 0.999

pca = PCA()
pca.fit(dataset.drop(columns=['A0']))    ### 如果做标准化，就用 data_scaled

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumulative_explained_variance >= num) + 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='.', label='Variance')
plt.axhline(y=num, color='r', linestyle='--', label=str(num))
plt.scatter(k, cumulative_explained_variance[k-1], color='g', s=100, zorder=5)
plt.annotate(f'k={k}', (k, cumulative_explained_variance[k-1]), 
             textcoords="offset points", xytext=(0, 20), ha='center', 
             fontsize=12, fontweight='bold', color='black')

plt.title('PCA')
plt.xlabel('k')
plt.ylabel('Variance')
plt.grid()
plt.legend()
plt.savefig("PCA.jpg")
plt.show()




res1 = []
res2 = []
res3 = []
res4 = []
res5 = []
res6 = []
res7 = []


resP = []
resR = []
resF = []


def unique_primes(start, n=10, k = 10 ):
    def pf(x):
        factors = set()
        while x % 2 == 0:
            factors.add(2)
            x //= 2
        i = 3
        while i*i <= x:
            while x % i == 0:
                factors.add(i)
                x //= i
            i += 2
        if x > 1:
            factors.add(x)
        return [ f for f in factors if f >= k-3 ]
    return sorted(set().union(*(pf(x) for x in range(start-n, start+1))))

# Example usage:
param_list = unique_primes(dataset_s.shape[1]-1, 10,k=k)
if k not in param_list:
    param_list = [k]+param_list
print(param_list)

for j in param_list:
    num = j
    s = setup(dataset, target = 'A0', session_id = 369, pca = True, pca_components = num)
    
    
    # abc = create_model('xgboost')  ################  xgboost,lightgbm,catboost #############
    
    # abc = create_model('xgboost', objective='binary:logitraw')  ################  不同的目标函数  #############
    
    abc = create_model('lightgbm', objective='xentlambda')  ################  不同的目标函数  #############
    
    # abc = create_model('catboost', objective='CrossEntropy')   ################  不同的目标函数  #############
    # compare_tree_models = compare_models(include = ['rf', 'xgboost', 'lightgbm', 'catboost'])

    
    abc_results = pull()
    abc_results = abc_results.loc[['Mean']]
    abc_results.to_csv('./temp/'+str(j)+f'r_{fix}.csv',index = False)
    
    final_best = finalize_model(abc)
    save_model(final_best, './temp/' + str(num) + 'x')
    data = pd.read_csv(test_path) #########  测试集文件  ########################
    data = data.drop(columns=['eob'])
    print(data.columns)
#     data.replace([np.inf, -np.inf], np.nan, inplace=True)   ####替换正负inf为NA
    
    predictions = predict_model(final_best, data=data) 
    
    n_preds = predictions['prediction_label'][num_xunlian:(num_xunlian+data_1_size)]    ### 取中间的数据
    n_preds = n_preds.reset_index(drop=True)                      ### 重置索引
    
    Note=open('./temp/' + str(num) + 'x.txt',mode='a')
    for i in range(0,data_1_size):         
        Note.write(str(n_preds[i]) + '\n') 
    Note.close()

    n_preds_score = predictions['prediction_score'][num_xunlian:(num_xunlian+data_1_size)]      ### 取中间的数据
    n_preds_score = n_preds_score.reset_index(drop=True)                   ### 重置索引
    
    Note=open('./temp/' + str(num) + 's.txt',mode='a')
    for i in range(0,data_1_size):         
        Note.write(str(n_preds_score[i]) + '\n') 
    Note.close()
    
    
    
    
    file_name ='./temp/Show.csv'
    df = pd.read_csv(file_name)
    path = './temp/'+str(j)+'x.txt'
    df2 = pd.read_csv(path, header=None, names=['state_x'])
    for i in range(0,data_1_size):  
        df['low'][i] = df2['state_x'][i]
   
    

    path = './temp/'+str(j)+'s.txt'
    df2 = pd.read_csv(path, header=None, names=['state_x'])
    df['score'] = 0
    for i in range(0,data_1_size):  
        df['score'][i] = df2['state_x'][i]
    df.to_csv('./temp/'+str(j)+f'x_{fix}.csv',index = False)
    
    
    
    
 

    file_name='./temp/'+  str(j) + f'x_{fix}.csv'
    data_1_new = pd.read_csv(file_name)

    aaa1 = data_1_new['volume']
    bbb1 = data_1_new['low']

    if buy == 0:
        for i in range(0,data_1_size):
            if bbb1.iloc[i] == 1:
                aaa1.iloc[i] = aaa1.iloc[i] * -1
    else:
        for i in range(0,data_1_size):
            if bbb1.iloc[i] == 0:
                aaa1.iloc[i] = aaa1.iloc[i] * -1

    for i in range(1,data_1_size):
        data_1_new['high'][i] = sum(data_1_new['volume'][0:(i+1)])

    data_1_new['high'][0] = data_1_new['volume'][0]

    for i in range(0,data_1_size):
        data_1_new['open'][i] = rrr * data_1_new['high'][i] / m


        
######################################################################################################


    wp_win = data_1_new['volume'] > 0
    wp_lost = data_1_new['volume'] < 0
    wp_nothing = data_1_new['volume'] == 0

    ### 满足条件的数量

    wp_win_a = wp_win.sum()            
    wp_lost_a = wp_lost.sum()
    wp_nothing_a = wp_nothing.sum()


    ### 满足条件的数据之和

    rrr_win = data_1_new[wp_win]['volume'].sum()
    rrr_lost = data_1_new[wp_lost]['volume'].sum()




    ##############################################################################################
    # 计算回撤数据，给到 down 列
    
    data_1_new['down'] = 0

    log = data_1_new['open'].iloc[0]

    for i in range(1,len(data_1_new)):

        if data_1_new['open'].iloc[i] < log:
            data_1_new['down'].iloc[i] = data_1_new['open'].iloc[i] - log
        else:
            log = data_1_new['open'].iloc[i]
        
    
    ##############################################################################################
    # 计算回撤面积，给到downarea列
    
    downarea = sum(data_1_new['down'])
    
    
    
    
    ##############################################################################################
    
    
    
    
    
    # 增加二级模型用到的列
    
    data_1_new['re'] = 0
    for i in range(1,len(data_1_new)):
        data_1_new['re'].iloc[i] = (data_1_new['close'].iloc[i] - data_1_new['close'].iloc[i-1]) / data_1_new['close'].iloc[i-1] * 100
        
    
    
    data_1_new['real'] = 0
    for i in range(1,len(data_1_new)):
        if data_1_new['close'].iloc[i] < data_1_new['close'].iloc[i-1]:
            data_1_new['real'].iloc[i] = 0
        else:
            data_1_new['real'].iloc[i] = 1
            
            
            
    data_1_new['real_lab'] = 'G'
    for i in range(1,len(data_1_new)):
        if buy == 0:
            if data_1_new['low'].iloc[i] != data_1_new['real'].iloc[i]:
                data_1_new['real_lab'].iloc[i] = 'G'
            else:
                data_1_new['real_lab'].iloc[i] = 'N'
        else:
            if data_1_new['low'].iloc[i] == data_1_new['real'].iloc[i]:
                data_1_new['real_lab'].iloc[i] = 'G'
            else:
                data_1_new['real_lab'].iloc[i] = 'N'
            
            
    file_name ='./temp/Show.csv'
    df = pd.read_csv(file_name)        
    data_1_new['show'] = df['low']
    
    
    
    data_1_new['show_lab'] = 'G'
    for i in range(1,len(data_1_new)):        
        if data_1_new['low'].iloc[i] == data_1_new['show'].iloc[i]:
            data_1_new['show_lab'].iloc[i] = 'G'
        else:
            data_1_new['show_lab'].iloc[i] = 'N'

        
      
    
    ##############################################################################################
    
    ### 计算夏普与索提诺
    
    data_1_new['re_real'] = 0
    for i in range(1,len(data_1_new)):
        if sell == 0:
            if data_1_new['low'].iloc[i] == 0:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i] * -1
            else:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i]
        else:
            if data_1_new['low'].iloc[i] == 1:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i] * -1
            else:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i]
    
    sharpe = round(data_1_new['re_real'][1:].mean() / data_1_new['re_real'][1:].std() * 100,4)
    
    sortino = round(data_1_new['re_real'][1:].mean() / (data_1_new['re_real'][1:][data_1_new['re_real'][1:] < 0]).std() * 100,4)
    
    ##############################################################################################
    
    
    
    
    
    
    data_1_new.to_csv('./temp/'+str(j)+f'x_{fix}.csv',index = False)
    
    
    
    
    s = np.argmax((np.maximum.accumulate(data_1_new['open']) - data_1_new['open'])) 
    if s == 0:
        e = 0
    else:
        e = np.argmax(data_1_new['open'][:s])  
    maxdrawdown = data_1_new['open'][e] - data_1_new['open'][s] # 最大回撤
    drawdown_days = s - e # 回撤持续周期数
    
    
    
    
    start_DAY = data_1_new.index[s] #开始回撤的日期
    end_DAY = data_1_new.index[e] #结束回撤的日期
    start_net_value = data_1_new[data_1_new.index == start_DAY]['open'].values[0] #开始回撤的净值
    end_net_value = data_1_new[data_1_new.index == end_DAY]['open'].values[0] #结束回撤的净值
    fig=plt.figure(figsize=(20,11))  
    plt.plot(data_1_new['eob'], data_1_new['open'])
    plt.plot([start_DAY, end_DAY], [start_net_value, end_net_value], linestyle='--', color='r')

    plt.xticks(range(0,data_1_size,int(data_1_size/m_size))) 

    plt.legend(['All:' + str(round(data_1_new['open'].iloc[-1]*100,2)) + '%' +
                '   ' + str(m_size) + 'm'
                '   Year:'+ str(round(data_1_new['open'].iloc[-1]/m_size*100*12,2)) + '%' +
                '   CalmarY:'+ str(round((data_1_new['open'].iloc[-1]/m_size*100*12)/(maxdrawdown*100),2)) +
                '   WP:' + str(round(wp_win_a/(wp_win_a + wp_lost_a)*100,2)) + '%' +
                '   RRR:' + str(round(rrr_win/(rrr_win+abs(rrr_lost))*100,2)) + '%' + ' / ' + str(round(rrr_win/abs(rrr_lost),2)) +
                '   T/N:' + str(wp_win_a + wp_lost_a ) + ' / ' + str(wp_nothing_a) +
                '   Sharpe:' + str(sharpe) +
                '   Sortino:' + str(sortino) +
                '   Accuracy:' + str(abc_results['Accuracy'][0]) +
                '   AUC:' + str(abc_results['AUC'][0]) +
                '   Recall:' + str(abc_results['Recall'][0]) +
                '   Prec:' + str(abc_results['Prec.'][0]) +
                '   F1:' + str(abc_results['F1'][0]) +
                '   Kappa:' + str(abc_results['Kappa'][0]) +
                '   MCC:' + str(abc_results['MCC'][0]),

                'MD:'+ str(round(maxdrawdown*100,2)) + '%' +
                '   DA:'+ str(round(downarea,4)) + '%' +
                '   MDT:' + str(drawdown_days)+
                '   Date:' + str(data_1_new['eob'].iloc[e]) + ' - ' + str(data_1_new['eob'].iloc[s])] ,

                loc='upper left',fontsize = 11)   ########### 默认是10
    
    
    plt.plot(data_1_new['eob'], data_1_new['down'], color='#ec700a')   ### 桔色
    plt.fill_between(data_1_new['eob'], data_1_new['down'], 0, where=(data_1_new['down']<0), facecolor='#FF0000', alpha=0.1)   
    plt.xticks(range(0,data_1_size,int(data_1_size/m_size)))                                           ### 红色 + 透明度
           

    
    fig.autofmt_xdate()
    plt.grid(1)
    plt.savefig("./temp/" + str(j) + "sy.jpg")
    plt.close()


    fig=plt.figure(figsize=(20,10))  
    plt.plot(data_1_new['eob'], data_1_new['high'])
    plt.xticks(range(0,data_1_size,int(data_1_size/m_size)))     ### 最后一个是间隔
    fig.autofmt_xdate()
    plt.grid(1)
    plt.savefig("./temp/" + str(j) + "p.jpg")
    plt.close()
    

    
    ##############################################################################################
    
    
    pp = abc_results['Prec.'][0]
    resP.append({
        'Prec_no': j,
        'max_Prec': pp
    })
    
    rr = abc_results['Recall'][0]
    resR.append({
        'Recall_no': j,
        'max_Recall': rr
    })
    
    ff = abc_results['F1'][0]
    resF.append({
        'F1_no': j,
        'max_F1': ff
    })
    

    
    
    ##############################################################################################
    
        

    max_all = round(data_1_new['open'].iloc[-1]*100,2)
    max_no = j

    res1.append({
        'All_no': max_no,
        'max_All': max_all
    })



    max_CalmarY = round((data_1_new['open'].iloc[-1]/m_size*100*12)/(maxdrawdown*100),2)
    
    res2.append({
        'CalmarY_no': max_no,
        'max_CalmarY': max_CalmarY
    })
    
    
    
    res3.append({
        'Downarea_no': max_no,
        'min_Downarea': downarea
    })
          
        
    max_wp = round(wp_win_a/(wp_win_a + wp_lost_a)*100,2)
    
    res4.append({
        'WP_no': max_no,
        'max_WP': max_wp
    })
    
    
    max_rrr = round(rrr_win/(rrr_win+abs(rrr_lost))*100,2)
    
    res5.append({
        'RRR_no': max_no,
        'max_RRR': max_rrr
    })
    
    
    res6.append({
        'Sharpe_no': max_no,
        'max_Sharpe': sharpe
    })
        
        
    res7.append({
        'Sortino_no': max_no,
        'max_Sortino': sortino
    })
    

   ##############################################################################################


aaaP = pd.DataFrame(resP)
aaaR = pd.DataFrame(resR)
aaaF = pd.DataFrame(resF)


bbbP = aaaP.sort_values(by="max_Prec",ascending=False)
bbbR = aaaR.sort_values(by="max_Recall",ascending=False)
bbbF = aaaF.sort_values(by="max_F1",ascending=False)


bbbP = bbbP.reset_index(drop=True)
bbbR = bbbR.reset_index(drop=True)
bbbF = bbbF.reset_index(drop=True)

bbbP['Recall_no'] = bbbR['Recall_no']
bbbP['max_Recall'] = bbbR['max_Recall']
bbbP['F1_no'] = bbbF['F1_no']
bbbP['max_F1'] = bbbF['max_F1']

bbbP.to_csv("./temp/Best_2.csv",index = False)


   ##############################################################################################



aaa1 = pd.DataFrame(res1)
aaa2 = pd.DataFrame(res2)
aaa3 = pd.DataFrame(res3)
aaa4 = pd.DataFrame(res4)
aaa5 = pd.DataFrame(res5)
aaa6 = pd.DataFrame(res6)
aaa7 = pd.DataFrame(res7)


bbb1 = aaa1.sort_values(by="max_All",ascending=False)       ### 由大到小排序
bbb2 = aaa2.sort_values(by="max_CalmarY",ascending=False)    
bbb3 = aaa3.sort_values(by="min_Downarea",ascending=False)     
bbb4 = aaa4.sort_values(by="max_WP",ascending=False)    
bbb5 = aaa5.sort_values(by="max_RRR",ascending=False)    
bbb6 = aaa6.sort_values(by="max_Sharpe",ascending=False)    
bbb7 = aaa7.sort_values(by="max_Sortino",ascending=False)   



bbb1 = bbb1.reset_index(drop=True)
bbb2 = bbb2.reset_index(drop=True)
bbb3 = bbb3.reset_index(drop=True)
bbb4 = bbb4.reset_index(drop=True)
bbb5 = bbb5.reset_index(drop=True)
bbb6 = bbb6.reset_index(drop=True)
bbb7 = bbb7.reset_index(drop=True)




bbb1['CalmarY_no'] = bbb2['CalmarY_no']
bbb1['max_CalmarY'] = bbb2['max_CalmarY']
bbb1['Downarea_no'] = bbb3['Downarea_no']
bbb1['min_Downarea'] = bbb3['min_Downarea']
bbb1['WP_no'] = bbb4['WP_no']
bbb1['max_WP'] = bbb4['max_WP']
bbb1['RRR_no'] = bbb5['RRR_no']
bbb1['max_RRR'] = bbb5['max_RRR']
bbb1['Sharpe_no'] = bbb6['Sharpe_no']
bbb1['max_Sharpe'] = bbb6['max_Sharpe']
bbb1['Sortino_no'] = bbb7['Sortino_no']
bbb1['max_Sortino'] = bbb7['max_Sortino']



bbb1.to_csv("./temp/Best_1.csv",index = False)









# %%



