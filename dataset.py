"""数据接口"""
import os
import torch
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    """pytorch规范，自定义数据集接口"""

    def __init__(self,path):
        self.eachFile(path)


    def __getitem__(self, index):
        inp = self.inps[index]
        score = self.scores[index]
        result = self.results[index]
        return inp, score,result


    def __len__(self):
        return self.inps.shape[0]


    def eachFile(self,filepath):
        """读取某个文件夹下的全部文件"""

        pathDir =  os.listdir(filepath)

        fens = []
        scores = []
        results = []

        for allDir in pathDir:
            child = os.path.join('%s/%s' % (filepath, allDir))
            fenst,scorest,resultst = self.readFile(child)
            fens = fens + fenst
            scores = scores + scorest
            results = results + resultst

        
        self.inps = torch.Tensor(fens)
        self.scores = torch.Tensor(scores)
        self.results = torch.Tensor(results)

        self.scores = torch.unsqueeze(self.scores,1)
        self.results = torch.unsqueeze(self.results,1)


    def readFile(self,filename):
        """读取单个文件"""
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


    def analysis(self,fenstr):
        """解析每一行，目前设定的是：己方设为1，无子设为0，对方设为-1，因此共有24个特征。
        另外，为了使得模型更好的训练，加入了己方在手，在棋盘上，对方在手，在棋盘上的棋子
        数目。因此共28个特征
        """

        ls1 = fenstr.split(" ")
        fen = []
        white = True
        for ch in ls1[0]:
            if ch == '*':
                fen.append(0)
            elif ch == '@':
                fen.append(-1)
            elif ch == 'O':
                fen.append(1)

        if ls1[1] == 'b':
            white = False
            for i , _ in enumerate(fen):
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
            if len(ls1[-1]) >= 6:
                result = 0.5
            elif len(ls1[-1]) <= 3:
                result = -1
            else :
                if ls1[-1][0] == '0' and ls1[-1][2] == '1':
                    result = 1
                elif ls1[-1][0] == '1' and ls1[-1][2] == '0':
                    result = 0
        else :
            if len(ls1[-1]) >= 4:
                result = 0.5
            elif len(ls1[-1]) <= 1:
                result = -1
            else :
                if ls1[-1][0] == '0' and ls1[-1][2] == '1':
                    result = 0
                elif ls1[-1][0] == '1' and ls1[-1][2] == '0':
                    result = 1
        return fen,score,result,white
