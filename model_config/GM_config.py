def create_model(model_class, args, print_info=False, **params):
    """
    创建模型实例的通用函数。
    
    :param model_class: 模型类（如 Model, Model_Intra, Model_Inter）
    :param args: 参数对象，包含所有必要的配置信息
    :param params: 参数字典，包含模型所需的所有维度形状信息 (如D_m, D_g等)
    :return: 创建的模型实例
    """
    params['print_info'] = print_info
    return model_class(
        args.base_model,
        params['D_text'],  # D_m
        params['D_g'],     # Graph feature dimension
        params['D_p'],     # Positional feature dimension
        params['D_e'],     # Emotional feature dimension
        params['D_h'],     # Hidden state feature dimension
        params['D_a'],     # Attention feature dimension
        params['graph_h'], # Graph hidden dimension
        n_speakers=params['n_speakers'],
        max_seq_len=200,
        window_past=args.windowp,
        window_future=args.windowf,
        n_classes=params['n_classes'],
        listener_state=args.active_listener,
        context_attention=args.attention,
        dropout=args.dropout,
        nodal_attention=args.nodal_attention,
        no_cuda=args.no_cuda,
        graph_type=args.graph_type,
        use_topic=args.use_topic,
        alpha=args.alpha,
        multiheads=args.multiheads,
        graph_construct=args.graph_construct,
        use_GCN=args.use_gcn,
        use_residue=args.use_residue,
        D_m_v=params['D_visual'],
        D_m_a=params['D_audio'],
        modals=args.modals,
        att_type=args.mm_fusion_mthd,
        av_using_lstm=args.av_using_lstm,
        Deep_GCN_nlayers=args.Deep_GCN_nlayers,
        dataset=args.Dataset,
        use_speaker=args.use_speaker,
        use_modal=args.use_modal,
        norm=args.norm,
        num_L=args.num_L,
        num_K=args.num_K,
        print_info = params['print_info'],
    )
