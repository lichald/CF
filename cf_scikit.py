#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: lining
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split

class CF:
    'user_based 基于scikit-learn的方法'
    
    def __init__(self):
        pass
    

    def recommend(self,df,rec_type='user_based'):
        
        if rec_type == 'user_based':
            #计算距离。余弦距离转换为余弦相似度，值越大越相关。因这个距离会作为权重来用
            similarity=1-pairwise_distances(df,metric='cosine',n_jobs=-1)
            similarityDF=pd.DataFrame(similarity,index=df.index,columns=df.index)
            #计算推荐的list,评分*权重（权重即是相似度）
            user_rec=np.dot(np.matrix(df).T,np.matrix(similarity))
            recDF=pd.DataFrame(user_rec,index=df.columns,columns=df.index)
    
        return recDF.T
    
    #计算precision和recall等交叉验证结果
    def crossValidation(self,trainDF,recDF,testDF):
        i=0
        for row in user_recDF.itertuples(index=True,name='Pandas'):
            #构建每个user的df，这里循环是第0位是index，从第1位开始是数值
            rec_per_user=pd.DataFrame({row[0]:row[1:]},index=user_recDF.columns)
            #对于推荐的结果降序排列
            rec_per_user.sort_values(by=row[0],inplace=True,ascending=False) 
            #取top 20 ，series转dataframe，每个user是一个列，转置成每个user是一行
            rec_per_user=rec_per_user.head(20).T 
            baseinfo=testDF.loc[[row[0],]] #基础信息
            test_columns=baseinfo.columns[list(set(np.where(baseinfo.notnull())[1]))]
            
            #构建结果
            dict1={"match":len(rec_per_user.columns & test_columns)
                    ,"rec":len(rec_per_user.columns),"actual":len(test_columns)}
            df1=pd.DataFrame(dict1,index=[row[0]])
            if i == 0 :
                self.cv_DF=df1
            else :
                self.cv_DF = self.cv_DF.append(df1)
            i += 1
        #计算precison和recall 
        self.cv_DF['precision']=self.cv_DF['match']/self.cv_DF['rec']
        self.cv_DF['recall']=self.cv_DF['match']/self.cv_DF['actual']
        
        
        
        
if __name__ == '__main__':
    #导入数据,small数据集，约600个人，9K个电影，10W评价
    df =  pd.read_csv('ratings.csv')
    df['rating']=df['rating']/5 #评分归一化
    
    #交叉验证，拆分样本
    trainDF,testDF =train_test_split(df,test_size=0.2,random_state=16)
    
    #行列重构，user_id为index，movie为column
    train_user_movie=trainDF.pivot(index='userId',columns='movieId',values='rating')
    train_user_movie.fillna(0,inplace=True)
    test_user_movie=testDF.pivot(index='userId',columns='movieId',values='rating')
    
    #基于用户做推荐
    cf=CF()
    user_recDF = cf.recommend(train_user_movie,rec_type='user_based')
    cf.crossValidation(train_user_movie,user_recDF,test_user_movie)
    print cf.cv_DF.describe()
