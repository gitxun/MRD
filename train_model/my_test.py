import numpy as np
import torch
from train_model.train_eval import train_or_eval_graph_model
from sklearn.metrics import confusion_matrix, f1_score


def test_model(args, model, model_mm, model_intra, model_inter, loss_function, test_loader, cuda):
    """测试模型"""
    print('Testing loaded model')

    # 运行测试，获取测试结果
    test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, log_probs = train_or_eval_graph_model(
        model, model_mm, model_intra, model_inter, args, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)

    # 打印测试准确率和 F-score
    print(f'Test Accuracy: {test_acc:.4f}, Test F-score: {test_fscore:.4f}')
    
    # 定义标签名称
    labels = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']

    # 打印混淆矩阵
    cm = confusion_matrix(test_label, test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # preds = []

    # # 对每个张量应用 softmax，并将结果追加到 softmax_outputs 列表中
    # for tensor in log_probs:
    #     preds.append(torch.argmax(tensor, 1).cpu().numpy())

    # if preds != []:
    #     preds  = np.concatenate(preds)
    # avg_fscore = round(f1_score(test_label,preds, average='weighted')*100, 2)
    # print(avg_fscore)
    
    return test_loss, test_acc, test_label, test_pred, test_fscore, log_probs
