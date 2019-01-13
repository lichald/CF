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

#导入数据,small数据集，600个人，9K个电影，10W评价
df =  pd.read_csv('ratings.csv')
df['rating']=df['rating']/5 #评分归一化
print df.head()
trainDF,testDF =train_test_split(df,test_size=0.2,random_state=16)
print trainDF.shape,testDF.shape
test_user_movie=testDF.pivot(index='userId',columns='movieId',values='rating')

def recommend(df):
    #整理出user_based的矩阵，（行是user，列是movie，内容是score）
    user_movie=df.pivot(index='userId',columns='movieId',values='rating')
    user_movie.fillna(0,inplace=True)

    #计算距离。余弦距离转换为余弦相似度，值越大越相关。因这个距离会作为权重来用
    similarity=1-pairwise_distances(user_movie,metric='cosine',n_jobs=-1)
    similarityDF=pd.DataFrame(similarity,index=user_movie.index,columns=user_movie.index)
    #计算推荐的list,评分*权重（权重即是相似度）
    user_rec=np.dot(np.matrix(user_movie).T,np.matrix(similarity))
    user_recDF=pd.DataFrame(user_rec,index=user_movie.columns,columns=user_movie.index)
    
    return user_recDF.T
'''
print '推荐userID=6的topN',user_recDF.loc[0:,6].sort_values(ascending=False).head(7)
print 'movieId=356的用户评分（汇总）',user_movie.loc[0:,356].sum()
print 'userID和其他人的相似度（汇总）',similarityDF.loc[0:,6].sum()
print '验证乘法是否正确userID=6,movieID=356 :',np.dot(similarityDF.loc[0:,6],user_movie.loc[0:,356])

'''
user_recDF=recommend(trainDF)
# 如何来对比两个df重叠度多少？
user_movie=df.pivot(index='userId',columns='movieId',values='rating')
i=0
for row in user_recDF.itertuples(index=True,name='Pandas'):
    print '='*10
    dict1={row[0]:row[1:]}
    df1=pd.DataFrame(dict1,index=user_recDF.columns)
    df1.sort_values(by=row[0],inplace=True,ascending=False) #降序排列
    df2=df1.head(20).T
    baseinfo=test_user_movie.loc[[row[0],]] #基础信息
    test_columns=baseinfo.columns[list(set(np.where(baseinfo.notnull())[1]))]
    print df1.describe() 
    print len(df2.columns & test_columns)
    print len(df2.columns)
    print len(test_columns)
    #都减去trainset的内容，然后计算precison和recall
    
    
    
    
    i +=1
    if i>0 :
        break


    
    
    
    
