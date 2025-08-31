import os
import time

import torch
from train_model.train_eval import train_or_eval_graph_model,train_or_eval_model
from data_load.data_loader import load_data
from utils.logging_util import log_metrics
from sklearn.metrics import  confusion_matrix, classification_report

def train_model(args, model, model_mm, model_intra, model_inter, loss_function, optimizer, n_epochs, batch_size, cuda, writer, best_mask):
    """训练模型"""
    best_fscore = None
    best_loss = None
    all_fscore = []

    for e in range(n_epochs):
        start_time = time.time()
        
        # 设置 epoch ratio
        epoch_ratio = min(1, round(args.epoch_ratio * int(e / args.scheduler_steps + 1), 2)) if args.courselearning else -1
        print("Current ratio =", epoch_ratio)

        # 加载数据
        train_loader, valid_loader, test_loader = load_data(args, batch_size, epoch_ratio)

        # 训练和评估
        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, _ = train_or_eval_graph_model(
                model, model_mm, model_intra, model_inter, args, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, model_mm, model_intra, model_inter, args, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _, _ = train_or_eval_graph_model(
                model, model_mm, model_intra, model_inter, args, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(
                model, args, cuda, loss_function, train_loader, e, writer, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(
                model, args, cuda, loss_function, valid_loader, e, writer)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(
                model, args, cuda, loss_function, test_loader, e, writer)

        all_fscore.append(test_fscore)

        # 更新最佳结果
        if best_loss is None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred
        
        if best_fscore is None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred = test_label, test_pred
            # 保存最佳模型
            best_model_path = os.path.join(args.save_dir, f'best_model_{args.Dataset}_epoch{e}.pkl')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model weights saved to {best_model_path}")

        # TensorBoard 日志
        if args.tensorboard:
            log_metrics(writer, train_acc, train_fscore, test_acc, test_fscore, e)

        # 打印训练信息
        print(f'Epoch: {e+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F-score: {train_fscore:.4f}\n '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F-score: {valid_fscore:.4f}\n'
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F-score: {test_fscore:.4f}\n'
              f'Time: {time.time()-start_time:.2f} sec')

        # 每10个epoch打印一次详细信息
        if (e + 1) % 5 == 0:
            print('Best F-Score:', max(all_fscore))
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

        # 早停条件：如果训练集的F分数达到95，则停止训练，因为已经过拟合
        if train_fscore >= 90:
            print("Early stopping as train F-score reached 90")
            break

    return best_fscore, best_label, best_pred
