import os
import time

import torch
from train_model.train_eval import train_or_eval_graph_model,train_or_eval_model,KD_model
from data_load.data_loader import load_data
from utils.logging_util import log_metrics
from sklearn.metrics import  confusion_matrix, classification_report

def train_model(args, model1, model2, loss_function, optimizer1, optimizer2, n_epochs, batch_size, cuda, writer, best_mask):
    """训练模型"""
    best_fscore1 = None
    best_fscore2 = None
    best_loss = None
    all_fscore1 = []
    all_fscore2 = []

    for e in range(n_epochs):
        start_time = time.time()
        
        # 设置 epoch ratio
        epoch_ratio = min(1, round(args.epoch_ratio * int(e / args.scheduler_steps + 1), 2)) if args.courselearning else -1
        print("Current ratio =", epoch_ratio)

        # 加载数据
        train_loader, valid_loader, test_loader = load_data(args, batch_size, epoch_ratio)

        # 训练和评估
        if args.graph_model:
            train_loss, _, _, _, train_acc1, train_acc2, train_fscore1, train_fscore2, _, _ = KD_model(
                model1, model2, args, loss_function, train_loader, e, cuda, args.modals, optimizer1, optimizer2, True, dataset=args.Dataset)
            valid_loss, _, _, _, valid_acc1, valid_acc2, valid_fscore1, valid_fscore2, _, _ = KD_model(
                model1, model2, args, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_label,test_pred1, test_pred2, test_acc1, test_acc2, test_fscore1, test_fscore2, _, _ = KD_model(
                model1, model2, args, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)

        all_fscore1.append(test_fscore1)
        all_fscore2.append(test_fscore2)

        # 更新最佳结果
        if best_loss is None or best_loss > test_loss:
            best_loss, best_label, best_pred1, best_pred2 = test_loss, test_label, test_pred1, test_pred2
        
        if best_fscore1 is None or best_fscore1 < test_fscore1:
            best_fscore1 = test_fscore1
            best_label, best_pred1 = test_label, test_pred1
            # 保存最佳模型
            best_model_path = os.path.join(args.save_dir, f'best_model1_{args.Dataset}_epoch{e}.pkl')
            torch.save(model1.state_dict(), best_model_path)
            print(f"Best model1 weights saved to {best_model_path}")
        if best_fscore2 is None or best_fscore2 < test_fscore2:
            best_fscore2 = test_fscore2
            best_label, best_pred2 = test_label, test_pred2
            # 保存最佳模型
            best_model_path = os.path.join(args.save_dir, f'best_model2_{args.Dataset}_epoch{e}.pkl')
            torch.save(model2.state_dict(), best_model_path)
            print(f"Best model2 weights saved to {best_model_path}")


        print(f'Epoch: {e+1}, Train Loss: {train_loss:.4f}, Train Acc1: {train_acc1:.4f}, Train F-score1: {train_fscore1:.4f}\n '
            #   f'Valid Loss: {valid_loss:.4f}, Valid Acc1: {valid_acc1:.4f}, Valid F-score1: {valid_fscore1:.4f}\n'
              f'Test Loss: {test_loss:.4f}, Test Acc1: {test_acc1:.4f}, Test F-score1: {test_fscore1:.4f}\n'
              f'Time: {time.time()-start_time:.2f} sec')
        print(f'Epoch: {e+1}, Train Loss: {train_loss:.4f}, Train Acc2: {train_acc2:.4f}, Train F-score2: {train_fscore2:.4f}\n '
            #   f'Valid Loss: {valid_loss:.4f}, Valid Acc2: {valid_acc2:.4f}, Valid F-score2: {valid_fscore2:.4f}\n'
              f'Test Loss: {test_loss:.4f}, Test Acc2: {test_acc2:.4f}, Test F-score2: {test_fscore2:.4f}\n'
              f'Time: {time.time()-start_time:.2f} sec')
        # 每10个epoch打印一次详细信息
        if (e + 1) % 1 == 0:
            print('Best F-Score1:', max(all_fscore1))
            print(classification_report(best_label, best_pred1, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred1, sample_weight=best_mask))
        if (e + 1) % 1 == 0:
            print('Best F-Score2:', max(all_fscore2))
            print(classification_report(best_label, best_pred2, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred2, sample_weight=best_mask))
        # 早停条件：如果训练集的F分数达到95，则停止训练，因为已经过拟合
        if train_fscore1 >= 95:
            print("Early stopping as train F-score1 reached 95")
            break

    return best_fscore1, best_fscore2, best_label, best_pred1, best_pred2
