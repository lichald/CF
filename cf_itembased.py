
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pandas import ExcelWriter
import time,datetime


class CF:
   
    
    def __init__(self):
        pass
    

    def recommend(self,df,rec_type='user_based',distance='cosine'):
        
        if rec_type == 'user_based':
            
            if distance == 'Cosine':
                similarity=1-pairwise_distances(df,metric='cosine',n_jobs=-1)
                similarityDF=pd.DataFrame(similarity,index=df.index,columns=df.index)
            
            if distance == 'Jaccard':
                similarity=1-pairwise_distances(df,metric='hamming',n_jobs=-1)
                similarityDF=pd.DataFrame(similarity,index=df.index,columns=df.index)
            
            user_rec=np.dot(np.matrix(df).T,np.matrix(similarity))/similarity.sum(axis=0)[np.newaxis,:]
            recDF=pd.DataFrame(user_rec,index=df.columns,columns=df.index).T

        if rec_type == 'item_based':
            if distance == 'Cosine':
                similarity=1-pairwise_distances(df.T,metric='cosine',n_jobs=-1)
                similarityDF=pd.DataFrame(similarity,index=df.columns,columns=df.columns)
            
            if distance == 'Adjusted-Cosine':
                df_NaN=df.replace(0, np.NaN)
                df_meanNaN = pd.concat([df_NaN.mean(axis=1)]*df_NaN.columns.size, axis=1)
                df_newNaN=pd.DataFrame(df_NaN.values-df_meanNaN.values,index=df_NaN.index,columns=df_NaN.columns)
                df_new=df_newNaN.replace(np.NaN,0)

                similarity=1-pairwise_distances(df_new.T,metric='cosine',n_jobs=-1)
                similarityDF=pd.DataFrame(similarity,index=df.columns,columns=df.columns)

            
            if distance == 'Jaccard':
                similarity=1-pairwise_distances(df.T,metric='hamming',n_jobs=-1)
                similarityDF=pd.DataFrame(similarity,index=df.columns,columns=df.columns)
            
            if distance == 'Pearson':
                similarity=df.corr(method='pearson', min_periods=1)
                similarityDF=pd.DataFrame(similarity,index=df.columns,columns=df.columns)
            
            item_rec=np.dot(np.matrix(df),np.matrix(similarity))/similarity.sum(axis=0)[np.newaxis,:]
            recDF=pd.DataFrame(item_rec,index=df.index,columns=df.columns)
        
       
        return recDF
    
    def crossValidation(self,trainDF,recDF,testDF,threshold=0,size=1000):
        i=0
        for row in recDF.itertuples(index=True,name='Pandas'):
            rec_per_user=pd.DataFrame({row[0]:row[1:]},index=recDF.columns)
            #rec_per_user.sort_values(by=row[0],inplace=True,ascending=False) 
            if threshold != 0:
                rec_per_user_special=rec_per_user[rec_per_user>threshold].T
            else :
                rec_per_user_special=rec_per_user.sort_values(by=[row[0]],ascending=False).head(size).T
            rec_columns=rec_per_user_special.columns[list(set(np.where(rec_per_user_special.notnull())[1]))]
        
            if str(row[0]) in testDF.index:
                baseinfo=testDF.loc[[row[0],]]
                test_columns=baseinfo.columns[list(set(np.where(baseinfo.notnull())[1]))]
                df_MAE=pd.concat([rec_per_user_special,baseinfo],axis=0).replace(np.NaN,0)
                mae=mean_absolute_error(df_MAE.iloc[0],df_MAE.iloc[1])
            else :
                test_columns=[-1] #invlid columns name
                mae=np.NaN
            
        
            dict1={"match":len(rec_columns & test_columns),"mae":mae
                    ,"rec":len(rec_columns),"actual":len(test_columns)}

            df1=pd.DataFrame(dict1,index=[row[0]])
            if i == 0 :
                self.cv_DF=df1
            else :
                self.cv_DF = self.cv_DF.append(df1)
            i += 1
        self.cv_DF['precision']=self.cv_DF['match']/self.cv_DF['rec']
        self.cv_DF['recall']=self.cv_DF['match']/self.cv_DF['actual']
        
        

def readfile(filename,row=-1):
    with open(filename) as f:
        data=[]
        i=0
        for line in f.readlines():
            item=line.strip().split('\t')
            data.append(item)
            i +=1
            if row>0 and i>row:
                break
        return data
    
    
def time_print(process,cur_time,last_time,start_time,length):
    time_show=datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d %H:%M:%S')
    cur_delta=int(cur_time-last_time)
    total_delta=int(cur_time-start_time)
    print('process:%s'%process,'cur_runtime=%d'%cur_delta,'toaltime=%d'%total_delta
                     ,'step=%d'%length)

#0<ratioFlag<1
def runCV(df,distanceMetricList,sizeList,ratioFlag=0,printProcess=True,writeFlag=False):
    k=0
    timelist=[time.time()]
    time_print('initiate',timelist[0],timelist[0],timelist[0],k)
    
    for ratio in np.arange(1,10,1):
        
        if ratioFlag > 1:
            continue
        elif ratioFlag>0:
            ratio=ratioFlag
            ratioFlag +=1
        else:            
            ratio = float(ratio) / 10

        trainDF,testDF =train_test_split(df,test_size=ratio)
        train_user_movie=trainDF.pivot(index='userid',columns='itemid',values='rating')
        train_user_movie.fillna(0,inplace=True)
        test_user_movie=testDF.pivot(index='userid',columns='itemid',values='rating')
    
        cf=CF()
        for distanceMetric in distanceMetricList:
        
            recDF = cf.recommend(train_user_movie,rec_type='item_based',distance=distanceMetric)

            for size in sizeList:


                cf.crossValidation(train_user_movie,recDF,test_user_movie,size=size)
                
                match=cf.cv_DF['match'].sum()
                actual=cf.cv_DF['actual'].sum()
                rec=cf.cv_DF['rec'].sum()
                mae=cf.cv_DF['mae'].mean()
                temp={'distanceMetric':distanceMetric,'ratio':ratio,'threshold':threshold,'size':size
                      ,'match':match,'rec':rec,'actual':actual,'mae':mae}      
                if actual != 0 and rec !=0 :
                    temp['accuracy']=float(match)/rec
                    temp['recall']=float(match)/actual
        
                if k == 0 :
                    tempDF=pd.DataFrame(temp,index=[k])
                    k +=1
                else :
                    tempDF = jieguo.append(pd.DataFrame(temp,index=[k]))
                    k +=1
                
                if printProcess:
                    timelist.append(time.time())
                    time_print('step',timelist[-1],timelist[-2],timelist[0],k)
    if writeFlag:
        writer = ExcelWriter('cf.xlsx')  
        tempDF.to_excel(writer,'cf') 
        writer.save() 
    else:
        return tempDF

if __name__ == '__main__':
    data=readfile('u.csv')
    df=pd.DataFrame(data,columns=['userid','itemid','rating','time_stamp'])
    df['rating']=df['rating'].apply(float)
    
    distanceMetricList=['Pearson','Cosine','Adjusted-Cosine','Jaccard']
    sizeList=[5,10,20,30,50]
    
    distanceMetricList2=['Jaccard']
    sizeList2=[10]
    cf=runCV(df,distanceMetricList2,sizeList2,ratioFlag=0.2)
    print cf
            
     



    
    
    
    
