import torch
import torch.optim as optim
from model_cm import  Model, Model_Inter, Model_Intra
import datetime
from data_load.data_loader import load_data   #data load
from args_setting import get_args     #args load
from utils.utils import seed_everything, ensure_dir_exists   #random_setting
from initialize_lossFunc import initialize_loss_function
from model_config import GM_config as gm, MM_config as mm, BM_config as bm, utils_config as config

from train_model.my_test import test_model
from train_model.my_train import train_model


if __name__ == '__main__':

    args = get_args()
    
    # 使用工具函数创建目录
    ensure_dir_exists(args.save_dir)

    today = datetime.datetime.now()
    print(args)

    # 调用 initialize_parameters 函数
    params,writer = config.initialize_parameters(args)
    if args.multi_modal:
        # 调用 initialize_multi_modal_parameters 函数
        params = mm.initialize_multi_modal_parameters(args, params)
    if args.graph_model:
        seed_everything(args.seed)

        model = gm.create_model(Model, args, **params, print_info=True)
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
        model.cuda()
        model_mm.cuda()
        model_intra.cuda()
        model_inter.cuda()

    loss_function = initialize_loss_function(args, cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)#Adam // AdamW // RAdam
    lr = args.lr

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    if args.testing:
        # 加载测试数据
        _, _, test_loader = load_data(args, batch_size)
        # 加载预训练模型
        if args.Dataset == 'MELD':
            state = torch.load("/data/lls/CMERC/MRD/saved_models/MELD_L.pkl")
        elif args.Dataset == 'IEMOCAP':
            state = torch.load("/data/lls/CMERC/MRD/saved_models/IEMOCAP_L.pkl")
        model.load_state_dict(state)
        test_loss, test_acc, test_label, test_pred, test_fscore, log_prob= test_model(args, model, model_mm, model_intra, model_inter,
                                                                             loss_function, test_loader, cuda)
    # 训练模型
    if not args.testing:
        best_fscore, best_label, best_pred = train_model(
            args, model, model_mm, model_intra, model_inter, loss_function, 
            optimizer, n_epochs, batch_size, cuda, writer, best_mask
        )
        print("Training completed. Best F-Score:", best_fscore)
