def create_class_weight_SCL(label):
    unique = list(set(label.cpu().detach().numpy().tolist()))

    # one = sum(label)
    labels_dict = {l:(label==l).sum().item() for l in unique}
    # labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for i in range(max(unique)+1):
        if i not in unique:
            weights.append(0)
        else:
            weights.append(total/labels_dict[i])
    return weights