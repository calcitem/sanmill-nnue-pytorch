"""模型训练"""

import logging
import time
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CreateDataset
from model import Module,QuantizeHelper
def parse_args():
    """配置"""
    parser = argparse.ArgumentParser(
                        description='nnue-module')
    # 需要将default后路径匹配
    parser.add_argument('--train', type=str, default="./data/data1",
                        help='训练集')
    # 这个暂时不用，目前的测试集是直接从训练集中分出来的。
    parser.add_argument('--val', type=str, default='./data',
                        help='测试集')
    # 学习率，过大会导致模型收敛不佳，过小使得模型的训练进度比较慢。推荐0.001-0.00001之间
    parser.add_argument('--lr', type=float, default='0.00001',
                        help='学习率')
    parser.add_argument('--use_quantize', action='store_true',
                        help='是否使用量化')
    parser.add_argument('--cpu', action='store_true',
                        help='在cpu上进行训练或检测')
    parser.add_argument('--is_test', action='store_true',
                        help='是否测试')
    # 若是之前已经训练好一个模型了，可以通过python main.py --from_checkpoint
    # 从上一次的训练结果继续训练。需要注意的是匹配好下面的best_modules_path的路径
    parser.add_argument("--from_checkpoint",action='store_true',help='是否从某一模型出发,\
                        若是请指定“best_modules_path”')
    # 模型参数的路径
    parser.add_argument('--best_modules_path', type=str, default="",\
                        help='random seed')
    parser.add_argument('--seed', type=int, default=0,\
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=2048,\
                        help='batch_size')
    # 训练轮数，推荐500-1000
    parser.add_argument('--epochs', type=int, default=1000,\
                        help='训练轮数')
    # 没过多少个epoch进行一次测试集验证
    parser.add_argument('--val_epochs_interval', type=int, default=5,\
                        help='验证步长')
    # 没多少个step打印一次训练的loss信息。
    parser.add_argument('--log_epochs_interval', type=int, default=30,\
                        help='每多少step打印log一次')
    parser.add_argument('--work_dir', type=str, default='./work_dir',\
                        help='文件输出路径')
    parser.add_argument('--gpu_id', type=int,default=0,\
                help='id of gpu to use (only applicable to non-distributed training)')
    args = parser.parse_args()
    return args
if __name__=="__main__":
    arg = parse_args()
    # 使神经网络的初始化每次都相同
    torch.manual_seed(arg.seed)

    work_dir = arg.work_dir
    # 创建对应参数的文件夹，记录实验数据。
    if not os.path.exists(arg.work_dir):
        os.mkdir(arg.work_dir)

    # 设置logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    work_dir += '/'+f'{timestamp}'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    #设置tensorboard
    train_writer = SummaryWriter(os.path.join(work_dir, 'train-loss'))
    val_writer = SummaryWriter(os.path.join(work_dir, 'val-loss'))

    if not os.path.exists(work_dir+"/logs"):
        os.mkdir(work_dir+"/logs")
    logging.basicConfig(filename=os.path.join(work_dir+"/logs", f'{timestamp}.log'), \
                            level=logging.DEBUG, format=LOG_FORMAT)

    if not os.path.exists(work_dir+"/best_modules"):
        os.mkdir(work_dir+"/best_modules")

    device = torch.device("cuda:{}".format(arg.gpu_id) if torch.cuda.is_available() else "cpu")

    if arg.cpu is True:
        device = torch.device("cpu")

    # 定义数据集
    mydataset = CreateDataset(arg.train)
    length=len(mydataset)

    # 将数据集以4:1比例分为训练集与测试集
    train_size , validate_size = int(0.8*length),length-int(0.8*length)
    train_set , validate_set = torch.utils.data.random_split(mydataset,[train_size,validate_size])
    train_loader = DataLoader(dataset=train_set,
                           batch_size=arg.batch_size,
                           shuffle=True)
    val_loader = DataLoader(dataset=validate_set,
                           batch_size=arg.batch_size,
                           shuffle=True)

    # 实例化网络
    mynet = Module().to(device)
    # 模型量化，初训练阶段暂时不要使用
    if arg.use_quantize is True:
        QuantizeHelper = QuantizeHelper()
        mynet = QuantizeHelper.get_prepared_module(mynet)
    logging.info("训练参数:\n",arg)

    # 从checkpoint恢复网络
    if arg.from_checkpoint is True:
        best_modules_path = arg.best_modules_path
        print("best model finished")
        logging.info("best model finished")
        mynet.load_state_dict(torch.load(best_modules_path))
    logging.info(mynet)
    logging.info("prepare model finished")
    print("prepare model finished")
    print("prepare_model:",mynet)
    # 定义损失函数
    loss_fn = torch.nn.MSELoss()

    # 定义adam优化器
    optimizer = torch.optim.Adam(mynet.parameters(), lr=arg.lr)

    # 记录最小验证精度
    min_val_mse = 10000
    min_loss_mse = 10000
    best_modules_path = ""
    best_modules_path1 = ""
    val_step = arg.val_epochs_interval

    # 开始训练
    for epoch in range(0,arg.epochs):
        print(epoch,"=========START=========")
        train_loss = 0

        #训练
        mynet.train()
        for i,batch_data in enumerate(train_loader):
            x,y,z = batch_data
            x = x.to(device)
            p = y.to(device)
            t = z.to(device)
            q = mynet(x)

            # MSELoss
            # 对应nnue的损失函数
            loss_eval = (p.sigmoid() - q.sigmoid()).square().mean()
            loss_result = (q - t).square().mean()
            loss = 0.6 * loss_eval + (1.0 - 0.6) * loss_result
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            st = "epoch: {}======step: {} ======loss: {}".format(epoch,i,loss.item())

            if i % arg.log_epochs_interval==0:
                print(st)
                logging.info(st)
        train_loss /= len(train_loader)

        # 验证
        if not epoch%val_step is 0:
            continue
        with torch.no_grad():
            val_loss = 0
            mynet.eval()
            for i,batch_data in enumerate(val_loader):
                x,y,z = batch_data
                x = x.to(device)
                y = y.to(device)
                output = mynet(x).sigmoid()
                loss = loss_fn(output,y.sigmoid())
                val_loss += loss.item()
            val_loss /= len(val_loader)

            if min_val_mse > val_loss:
                min_val_mse = val_loss
                logging.info("min val_mse:{}".format(min_val_mse))
                best_modules_path = work_dir+"/best_modules/"+'best-'+timestamp+'.pt'
                torch.save(mynet.state_dict(), best_modules_path)
                # if arg.use_quantize is True:
                #     mynet_quantized = QuantizeHelper.get_quantized_module(mynet)
                #     best_qmodules_path = work_dir+"/best_modules/"+'best-'+timestamp+'qmodule.pt'
                #     torch.save(mynet_quantized.state_dict(), best_qmodules_path)

            if min_loss_mse > train_loss:
                min_loss_mse = train_loss
                logging.info("min train_loss_mse:{}".format(min_loss_mse))
                best_modules_path1 = work_dir+"/min_trainloss_modules/"+'best-'+timestamp+'.pt'
                torch.save(mynet.state_dict(), best_modules_path1)
            st = "| val:epoch: {} |  train_loss: {} |val_loss: {}".format(epoch,train_loss,val_loss)
            logging.info(st)
            print(st)

        train_writer.add_scalar("mse", train_loss, epoch+1)
        val_writer.add_scalar("mse", val_loss, epoch+1)

    logging.info("min val_mse:{}".format(min_val_mse))
    print("min val_mse:{}".format(min_val_mse))

    # 测试，这个在训练阶段暂时不要考虑
    # 加载最优模型
    if arg.is_test is True:
        mynet.load_state_dict(torch.load(best_modules_path))
        mynet.eval()

        with torch.no_grad():
            test_loss=0

            for data in val_loader:
                x,y = data
                x = x.to(device)
                y = y.to(device)
                predict = mynet(x).sigmoid()
                loss = loss_fn(predict,y.sigmoid())
                test_loss+=loss.item()

            test_loss /= len(val_loader)
            print("test_loss:",test_loss)
            logging.info("test_loss:{}".format(test_loss))
