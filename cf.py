#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:10:09 2019

@author: administrator
"""
import sys,time,datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

def time_print(process,cur_time,last_time,start_time,length):
    time_show=datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d %H:%M:%S')
    cur_delta=int(cur_time-last_time)
    total_delta=int(cur_time-start_time)
    print('process:%s'%process,'cur_runtime=%d'%cur_delta,'toaltime=%d'%total_delta,'length=%d'%length)


class CF:
    '协同过滤：user_base和item_base'
    
    def __init__(self, movies,ratings):
        self.movies = movies
        self.ratings = ratings
        

     #第一步 变成DF准备 是针对用户和商品新建表,user+movie+评分
    def formatDF(self):     
        userDict={}
        itemDict={}
        
        for rating in self.ratings:  
            temp1={}
            temp2={}
            #temp1[int(rating[1])]=float(rating[2])/5    
            if int(rating[0]) not in userDict.keys():
                userDict[int(rating[0])]={int(rating[1]):float(rating[2])/5}
            else :
                userDict[int(rating[0])][int(rating[1])]=float(rating[2])/5
            
            
            #temp2[int(rating[0])]=float(rating[2])/5
            if int(rating[1]) not in itemDict.keys():
                itemDict[int(rating[1])]={int(rating[0]):float(rating[2])/5}
            else:
                itemDict[int(rating[1])][int(rating[0])]=float(rating[2])/5
        
        self.userDictDF=userDict
        self.itemDictDF=itemDict
        
        print 'list2dict完成'
        i=0
        for k,v in  self.userDictDF.iteritems():
            data=pd.DataFrame(v,index=[k])
            if i == 0:
                i +=1
                self.userDF=data
            else:
                self.userDF=pd.concat([self.userDF,data])

        
        i=0
        for k,v in  self.itemDictDF.iteritems():
            data=pd.DataFrame(v,index=[k])
            if i == 0:
                i +=1
                self.itemDF=data
            else:
                self.itemDF=pd.concat([self.itemDF,data])
        
        print 'dict2DF完成'

    #第二步 找到该人与所有的距离，把别人的电影*距离赋给你
    def recommendList(self):
        #填充异常值
        self.userDF.fillna(0,inplace=True)   
        #计算余弦距离（并将距离index加上）
        dist=cosine_similarity(self.userDF) 
        distDF=pd.DataFrame(dist,index=self.userDF.index,columns=self.userDF.index)
        
        #DF转为矩阵，计算权重（距离即为权重）加和
        userMat=np.matrix(self.userDF.values)
        distMat=np.matrix(distDF.values)
        print userMat.T.shape,distMat.shape
        outcome=np.dot(userMat.T , distMat)
        outcomeDF=pd.DataFrame(outcome,index=self.userDF.columns,columns=self.userDF.index)
        
        print outcomeDF[7].sort_values(ascending=False).head(n=10)
        '''
        outcomeDF.to_csv('cf_jieguo1.csv')
        distDF.to_csv('cf_distDF1.csv')
        self.userDF.to_csv('cf_userDF1.csv')
        '''
        
        
        
def readfile(filename,row=-1):
    with open(filename) as f:
        data=[]
        i=0
        for line in f.readlines():
            item=line.strip().split('::')
            data.append(item)
            i +=1
            if row>0 and i>row:
                break
        return data

#movie有10628部，评分有10million。这里面限制row来做测试
movies=readfile('movies.dat')
ratings=readfile('ratings.dat')
initiate=time.time()
time_print('initiate',initiate,initiate,initiate,len(ratings))

cf_sample=CF(movies,ratings)
load=time.time()
time_print('load',load,initiate,initiate,len(cf_sample.movies)+len(cf_sample.ratings))

cf_sample.formatDF()
list2dict=time.time()
time_print('list2dict',list2dict,load,initiate,0)


cf_sample.recomendList()
dict2df=time.time()
time_print('dict2df',dict2df,list2dict,initiate,0)

