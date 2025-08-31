import torch
from utils.logging_util import initialize_tensorboard

def initialize_parameters(args):
    """
    根据输入的 args 参数，初始化通用的模型和训练参数。
    """

    # 设置模型名称
    if args.av_using_lstm:
        name_ = (
            f"{args.mm_fusion_mthd}_{args.modals}_{args.graph_type}_"
            f"{args.graph_construct}using_lstm_{args.Dataset}"
        )
    else:
        name_ = (
            f"{args.mm_fusion_mthd}_{args.modals}_{args.graph_type}_"
            f"{args.graph_construct}{args.Deep_GCN_nlayers}_{args.Dataset}"
        )

    if args.use_speaker:
        name_ += "_speaker"
    if args.use_modal:
        name_ += "_modal"

    # 检查是否使用 GPU
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print("Running on GPU")
    else:
        print("Running on CPU")

    # 如果启用了 TensorBoard，初始化相关工具
    writer = None
    if args.tensorboard:
        writer = initialize_tensorboard()  # 假设函数 initialize_tensorboard 已定义

    # 获取基本参数
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    

    # 定义特征维度映射
    feat2dim = {
        "IS10": 1582,
        "3DCNN": 512,
        "textCNN": 100,
        "bert": 768,
        "denseface": 342,
        "MELD_text": 600,
        "MELD_audio": 300,
    }

    # 根据数据集选择特征维度
    D_audio = feat2dim["IS10"] if args.Dataset == "IEMOCAP" else feat2dim["MELD_audio"]
    D_visual = feat2dim["denseface"]
    D_text = 1024  # 固定值，也可以动态调整->feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    # 其他固定参数
    D_g = 512 if args.Dataset == "IEMOCAP" else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset == "MELD" else 2
    n_classes = 7 if args.Dataset == "MELD" else 6 if args.Dataset == "IEMOCAP" else 1

    # 返回所有初始化的参数
    return {
        "name_": name_,
        "writer": writer,
        "cuda": cuda,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "modals": modals,
        "D_audio": D_audio,
        "D_visual": D_visual,
        "D_text": D_text,
        "D_g": D_g,
        "D_p": D_p,
        "D_e": D_e,
        "D_h": D_h,
        "D_a": D_a,
        "graph_h": graph_h,
        "n_speakers": n_speakers,
        "n_classes": n_classes,
    },writer