from model_cm import GRUModel,LSTMModel
def create_model(args, **params):
    """
    创建模型实例的通用函数。
    
    :param args: 参数对象，包含所有必要的配置信息
    :param params: 参数字典，包含模型所需的所有维度形状信息 (如D_m, D_g等)
    :return: 创建的模型实例
    """
    D_m, D_e, D_h = params['D_text'], params['D_e'], params['D_h']
    n_classes = params['n_classes']

    if args.base_model == 'GRU':
        model = GRUModel(D_m, D_e, D_h, 
                         n_classes=n_classes, 
                         dropout=args.dropout)
        print('Basic GRU Model.')
    
    elif args.base_model == 'LSTM':
        model = LSTMModel(D_m, D_e, D_h, 
                          n_classes=n_classes, 
                          dropout=args.dropout)
        print('Basic LSTM Model.')
    else:
        print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
        raise NotImplementedError
    return model