"""模型的具体结构"""
import copy

import torch
from torch.quantization import quantize_fx as quantize_fx

# 参数量通过这三个进行控制。数值越大，参数量越大。
L1 = 64
L2 = 32
L3 = 16


class Module(torch.nn.Module):
    """nnue模型
    由于直棋的特征数远远小于象棋的特征数，使得我们可以适当加大网络非首层的参数量，
    因此增加了fc21，fc22两层。而为了模型的训练，添加了bn层。
    """

    def __init__(self):
        super(Module, self).__init__()
        # 若是后期改变数据的特征数目，将28换位对应的数值即可。
        self.fc1 = torch.nn.Linear(28,L1,bias=True)
        self.bn1 = torch.nn.BatchNorm1d(L1,eps=1e-05,momentum=0.1,affine=True,\
                                            track_running_stats=True)

        self.fc2 = torch.nn.Linear(L1,L1,bias=True)
        self.bn2 = torch.nn.BatchNorm1d(L1, eps=1e-05, momentum=0.1, affine=True,\
             track_running_stats=True)

        # self.fc21 = torch.nn.Linear(L1,L1,bias=True)
        # self.bn21 = torch.nn.BatchNorm1d(L1, eps=1e-05, momentum=0.1, affine=True,\
        #      track_running_stats=True)

        # self.fc22 = torch.nn.Linear(L1,L1,bias=True)
        # self.bn22 = torch.nn.BatchNorm1d(L1, eps=1e-05, momentum=0.1, affine=True, \
        #     track_running_stats=True)

        self.fc3 = torch.nn.Linear(L1,L2,bias=True)
        self.bn3 = torch.nn.BatchNorm1d(L2, eps=1e-05, momentum=0.1, affine=True,\
             track_running_stats=True)

        self.fc4 = torch.nn.Linear(L2,L3,bias=True)
        self.bn4 = torch.nn.BatchNorm1d(L3, eps=1e-05, momentum=0.1, affine=True,\
             track_running_stats=True)

        self.fc5 = torch.nn.Linear(L3,1,bias=True)

        self.relu = torch.nn.ReLU()


    def forward(self,x):
        """模型前向传播函数"""
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

       # out = self.fc21(out)
        #out = self.bn21(out)
       # out = self.relu(out)

        #out = self.fc22(out)
        #out = self.bn22(out)
        #out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.fc5(out)
        return out


    def trace_module(self,path=".",shape=(1,28)):
        """转换为script_module"""
        input1 = torch.rand(shape)
        self.eval()
        script_module = torch.jit.trace(self,input1)
        torch.jit.save(script_module, path+"/script_model.pt")


    def load_script_module(self,path):
        """加载script-module"""
        model=torch.jit.load(path)
        return model


class QuantizeHelper():
    """模型量化助手类"""
    def __init__ (self):
        pass


    def get_prepared_module(self,model_fp):
        """得到一个可训练的量化模型"""
        model_to_quantize = copy.deepcopy(model_fp)
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
        model_to_quantize.train()
        # prepare
        model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_dict)
        return model_prepared


    def get_quantized_module(self,prepared_model):
        """将模型转化为真量化模型"""
        model_to_quantize = copy.deepcopy(prepared_model)
        model_to_quantize.eval()
        model_quantized = quantize_fx.convert_fx(model_to_quantize)
        return model_quantized
