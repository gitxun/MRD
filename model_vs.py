import torch
import torch.optim as optim
from model_cm import Model, Model_Inter, Model_Intra
import datetime
from data_load.data_loader import load_data   # data load
from args_setting import get_args   # args load
from utils.utils import seed_everything, ensure_dir_exists   # random_setting
from initialize_lossFunc import initialize_loss_function
from model_config import GM_config as gm, MM_config as mm, BM_config as bm, utils_config as util

from train_model.my_test import test_model


# def compare_results(test_label, predictions1, predictions2):
#     # 计算权重1预测错误但权重2预测正确的样本数量
#     incorrect1_correct2 = ((predictions1 != test_label) & (predictions2 == test_label)).sum().item()
#     # 计算权重1预测正确但权重2预测错误的样本数量
#     correct1_incorrect2 = ((predictions1 == test_label) & (predictions2 != test_label)).sum().item()
#     return incorrect1_correct2, correct1_incorrect2
def compare_results(test_label, predictions1, predictions2):
    # 计算权重1预测错误但权重2预测正确的样本布尔索引
    mask_incorrect1_correct2 = (predictions1 != test_label) & (predictions2 == test_label)
    # 计算权重1预测正确但权重2预测错误的样本布尔索引
    mask_correct1_incorrect2 = (predictions1 == test_label) & (predictions2 != test_label)

    # 统计数量
    incorrect1_correct2 = mask_incorrect1_correct2.sum().item()
    correct1_incorrect2 = mask_correct1_incorrect2.sum().item()

    # 统计对应标签的数量，使用字典统计
    from collections import Counter

    incorrect1_correct2_labels = Counter(test_label[mask_incorrect1_correct2].tolist())
    correct1_incorrect2_labels = Counter(test_label[mask_correct1_incorrect2].tolist())

    return incorrect1_correct2, correct1_incorrect2, incorrect1_correct2_labels, correct1_incorrect2_labels



if __name__ == '__main__':
    args = get_args()
    ensure_dir_exists(args.save_dir)

    today = datetime.datetime.now()
    print(args)

    params, writer = util.initialize_parameters(args)
    if args.multi_modal:
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    if args.testing:
        _, _, test_loader = load_data(args, batch_size)

        # Define paths to your weight files
        weight_files = [
            "/data/lls/CMERC/MRD/saved_models/MELD_checkpoint.pkl",
            "/data/lls/CMERC/MRD/saved_models/MELD_L.pkl"
        ]
        
        all_predictions = []
        
        for weight_file in weight_files:
            state = torch.load(weight_file)
            model.load_state_dict(state)
            _, _, test_label, test_pred, _, _ = test_model(args, model, model_mm, model_intra, model_inter,
                                                         loss_function, test_loader, cuda)
            all_predictions.append(test_pred)
        
        # Compare results between the first two weight files
        # incorrect1_correct2, correct1_incorrect2 = compare_results(test_label, all_predictions[0], all_predictions[1])
        # print(f"Incorrect1 & Correct2: {incorrect1_correct2}, Correct1 & Incorrect2: {correct1_incorrect2}")
        

        incorrect1_correct2, correct1_incorrect2, incorrect_labels, correct_labels = compare_results(test_label, all_predictions[0], all_predictions[1])
        print("权重1错权重2对的样本数:", incorrect1_correct2)
        print("对应标签分布:", incorrect_labels)
        print("权重1对权重2错的样本数:", correct1_incorrect2)
        print("对应标签分布:", correct_labels)

