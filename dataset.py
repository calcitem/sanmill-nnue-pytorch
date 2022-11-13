

import torch
from model import Module1
from torch.utils.data import DataLoader,Dataset
import logging
import argparse
import os
class CreateDataset(Dataset):
    def __init__(self,path,data_size=None):
        self.eachFile(path)
    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        z1 = self.z[index]
        return x1, y1,z1
    def __len__(self):
        return self.x.shape[0]
    ###读取某个文件夹下的全部文件
    def eachFile(self,filepath):
        pathDir =  os.listdir(filepath)
        fens= []
        scores=[]
        results=[]
        for allDir in pathDir:
            child = os.path.join('%s/%s' % (filepath, allDir))
            fenst,scorest,resultst = self.readFile(child)
            fens = fens + fenst
            scores = scores + scorest
            results = results + resultst
        self.x = torch.Tensor(fens)
        self.y = torch.Tensor(scores)
        self.z = torch.Tensor(results)
        self.y = torch.unsqueeze(self.y,1)
        self.z = torch.unsqueeze(self.z,1)
        return self.x,self.y,self.z        
    ##读取单个文件
    def readFile(self,filename):
        fopen = open(filename, 'r') 
        fens=[]
        scores=[]
        results=[]
        whites = []
        result1 = 0
        for eachLine in fopen:
            if len(eachLine)>=20:
                fen,score,result ,white= self.analysis(eachLine)
                result1=result
                fens.append(fen),scores.append(score),whites.append(white)
        for i in range(0,len(fens)):
            if whites[i] == whites[-1]:
                results.append(result1)
            else :
                results.append(1-result)
        fopen.close()
        return fens,scores,results
    ###解析每一行，目前设定的是：己方设为1，无子设为0，对方设为-1，因此共有24个特征。
    ###另外，为了使得模型更好的训练，加入了己方在手，在棋盘上，对方在手，在棋盘上的棋子数目。因此共28个特征
    def analysis(self,fenstr):
        # **@*O*@*/*@**O@@*/****O@@* b m r 3 0 7 0 1 1 1 -91 (2,8)->(2,7) 48 0-1
        ls1 = fenstr.split(" ")
        fen = []
        white = True
        for ch in ls1[0]:
            if ch == '*':
                fen.append(0)
            elif ch =='@':
                fen.append(-1)
            elif ch=='O':
                fen.append(1)
        if ls1[1]=='b':
            white = False
            for i,_ in enumerate(fen):
                fen[i] = 0-fen[i]
            fen.append(float(ls1[6]))
            fen.append(float(ls1[7]))
            fen.append(float(ls1[4]))
            fen.append(float(ls1[5]))
        else:
            fen.append(float(ls1[4]))
            fen.append(float(ls1[5]))
            fen.append(float(ls1[6]))
            fen.append(float(ls1[7]))
            

        score = int(ls1[-4])
        result = 0
        if ls1[1] == 'b':
            if len(ls1[-1])>=6:
                result = 0.5
            elif len(ls1[-1])<=3:
                result = -1
            else :
                if ls1[-1][0]=='0' and ls1[-1][2]=='1':
                    result = 1
                elif ls1[-1][0]=='1' and ls1[-1][2]=='0':
                    result = 0
        else :
            if len(ls1[-1])>=4:
                result = 0.5
            elif len(ls1[-1])<=1:
                result = -1
            else :
                if ls1[-1][0]=='0' and ls1[-1][2]=='1':
                    result = 0
                elif ls1[-1][0]=='1' and ls1[-1][2]=='0':
                    result = 1
        return fen,score,result,white 


# if __name__ == "__main__":
#     dataset = CreateDataset("/home/data/qxh/code/zyy_dlworks/test/data/data1")
#     a,b,c = dataset.eachFile("/home/data/qxh/code/zyy_dlworks/test/data/data1")
    # a,b,c = dataset.readFile("/home/data/qxh/code/zyy_dlworks/test/data/data1/training-data_s1q4.0_1668211242.txt")
    # a,b,c = dataset.analysis("**@*O*@*/*@**O@@*/****O@@* b m r 3 0 7 0 1 1 1 -91 (2,8)->(2,7) 48 1/2-1/2")
    # print(a,"\n",b,"\n",c)    
    # print(len(a),"\n",len(b),"\n",len(c))
    # print(type(a),"\n",type(b),"\n",type(c))
    # print(a.shape,"\n",b.shape,"\n",c.shape)  