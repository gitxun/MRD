def initialize_multi_modal_parameters(args, params):
    """
    根据多模态相关的设置，初始化多模态相关的参数。
    """
    D_audio = params['D_audio']
    D_visual = params['D_visual']
    D_text = params['D_text']
    modals = params['modals']

    # 根据多模态融合方法计算 D_m
    if args.multi_modal:
        if args.mm_fusion_mthd == "concat":
            if modals == "avl":
                D_m = D_audio + D_visual + D_text
            elif modals == "av":
                D_m = D_audio + D_visual
            elif modals == "al":
                D_m = D_audio + D_text
            elif modals == "vl":
                D_m = D_visual + D_text
            else:
                raise NotImplementedError(f"Unsupported modals: {modals}")
        else:
            D_m = 1024  # 默认维度
    else:
        if modals == "a":
            D_m = D_audio
        elif modals == "v":
            D_m = D_visual
        elif modals == "l":
            D_m = D_text
        else:
            raise NotImplementedError(f"Unsupported modals: {modals}")

    # 添加多模态参数到 params 中
    params['D_m'] = D_m

    return params
