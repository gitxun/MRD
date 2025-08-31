import torch
import numpy as np
import torch.optim as optim
from model_cm import Model, Model_Inter, Model_Intra
import datetime
from data_load.data_loader import load_data  # Data loading
from args_setting import get_args  # Arguments loading
from utils.utils import seed_everything, ensure_dir_exists  # Random setting
from initialize_lossFunc import initialize_loss_function
from model_config import GM_config as gm, MM_config as mm, BM_config as bm, utils_config as config
from train_model.kd_train import train_model


if __name__ == '__main__':
    args = get_args()
    
    # 使用工具函数创建目录
    ensure_dir_exists(args.save_dir)

    today = datetime.datetime.now()
    print(args)

    # 调用 initialize_parameters 函数
    params, writer = config.initialize_parameters(args)
    if args.multi_modal:
        params = mm.initialize_multi_modal_parameters(args, params)
    if args.graph_model:
        seed_everything(args.seed)

        model1 = gm.create_model(Model, args, **params, print_info=True)
        model2 = gm.create_model(Model, args, **params)
        model_mm = gm.create_model(Model, args, **params)
        model_intra = gm.create_model(Model_Intra, args, **params)
        model_inter = gm.create_model(Model_Inter, args, **params)

        print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'
    else:
        bm.create_model(args, **params)
        name = 'Base'

    cuda = params['cuda']
    if cuda:
        model1.cuda()
        model2.cuda()
        model_mm.cuda()
        model_intra.cuda()
        model_inter.cuda()

    loss_function = initialize_loss_function(args, cuda)
    # 为每个模型定义优化器
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.l2)
    lr = args.lr
    
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    
    # 加载预训练模型
    if args.Dataset == 'MELD':
        state_1 = torch.load("/data/lls/CMERC/MRD/saved_models/MELD_checkpoint.pkl")
        state_2 = torch.load("/data/lls/CMERC/MRD/saved_models/MELD_L.pkl")
        model1.load_state_dict(state_1)  # 加载 model1 的权重
        model2.load_state_dict(state_2)  # 加载 model2 的权重
    elif args.Dataset == 'IEMOCAP':
        state_1 = torch.load("/data/lls/CMERC/MRD/saved_models/IEMOCAP_checkpoint.pkl")
        state_2 = torch.load("/data/lls/CMERC/MRD/saved_models/IEMOCAP_L.pkl")
        model1.load_state_dict(state_1)  # 加载 model1 的权重
        model2.load_state_dict(state_2)  # 加载 model2 的权重
    #start train    
    best_fscore1, best_fscore2, _, best_pred1, best_pred2 = train_model(
        args, model1, model2, loss_function, optimizer1, optimizer2, n_epochs, batch_size, cuda, writer, best_mask)
    print("Training completed. Best F-Score1:", best_fscore1)
    print("Training completed. Best F-Score2:", best_fscore2)