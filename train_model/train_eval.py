import numpy as np
import torch
from tqdm import tqdm
from args_setting import get_args     #args load
from utils.utils import seed_everything   #random_setting
from class_weight import create_class_weight_SCL
from sklearn.metrics import f1_score, accuracy_score
from clloss import CKSCL_context, CKSCL_tav
import torch.nn.functional as F

CKSCL_m = CKSCL_tav()
CKSCL_c = CKSCL_context()

import torch.nn as nn


def attention_weight_from_confidence(log_prob1, log_prob2):
    # 计算两个模型的最大置信度
    prob1_max = torch.max(F.softmax(log_prob1, dim=1), dim=1)[0]
    prob2_max = torch.max(F.softmax(log_prob2, dim=1), dim=1)[0]
    # 计算总置信度
    total_confidence = prob1_max + prob2_max
    weight1 = prob1_max / total_confidence
    weight2 = prob2_max / total_confidence
    return weight1


def train_or_eval_model(model, args, cuda, loss_function, dataloader, epoch, writer, optimizer=None, train=False):
    """
    
    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0))
        
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, model_mm, model_intra, model_inter, args, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    model_mm.load_state_dict(model.state_dict())
    model_mm.eval()
    model_intra.load_state_dict(model.state_dict())
    model_intra.eval()
    model_inter.load_state_dict(model.state_dict())
    model_inter.eval()

    seed_everything(args.seed)
    # 新增一个列表来存储所有批次的 log_prob
    all_log_probs = []
    
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
                
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                # 初始化 textf，假设是将多个文本特征拼接在一起
                textf = torch.cat([textf1, textf2, textf3, textf4], dim=-1)
            else:
                raise NotImplementedError
        # 1. Cross Entropy Loss
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        if args.contrastlearning or args.calibrate:
            with torch.no_grad():
                ft_scl = hidden
                mask_textf1 = torch.mean(textf1, dim=0).repeat(textf1.shape[0], 1, 1)
                mask_textf2 = torch.mean(textf2, dim=0).repeat(textf2.shape[0], 1, 1)
                mask_textf3 = torch.mean(textf3, dim=0).repeat(textf3.shape[0], 1, 1)
                mask_textf4 = torch.mean(textf4, dim=0).repeat(textf4.shape[0], 1, 1)

                mask_acouf = torch.mean(acouf, dim=0).repeat(acouf.shape[0], 1, 1)
                mask_visuf = torch.mean(visuf, dim=0).repeat(visuf.shape[0], 1, 1)
                
                # mask multimodal data
                log_prob2t, e_i, e_n, e_t, e_l, hiddent = model_mm([mask_textf1, mask_textf2, mask_textf3, mask_textf4], qmask, umask, lengths, acouf, visuf, epoch)
                log_prob2a, e_i, e_n, e_t, e_l, hiddena = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, mask_acouf, visuf, epoch)
                log_prob2v, e_i, e_n, e_t, e_l, hiddenv = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, mask_visuf, epoch)

                # mask inter context and intra contex
                log_prob2intra, e_i, e_n, e_t, e_l, hidden_intra = model_intra([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
                log_prob2inter, e_i, e_n, e_t, e_l, hidden_inter = model_inter([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)


                labels_ = label

                pred2t_ = torch.argmax(log_prob2t, 1) # batch*seq_len
                pred2a_ = torch.argmax(log_prob2a, 1) # batch*seq_len
                pred2v_ = torch.argmax(log_prob2v, 1) # batch*seq_len
                pred2c_intra = torch.argmax(log_prob2intra, 1)
                pred2c_inter = torch.argmax(log_prob2inter, 1)

            # 2. rank loss, indicate cf
            lp_ = log_prob
            pred_ = torch.argmax(lp_,1)
            rank_loss = 0
            for i in range(len(log_prob)):
                num = labels_[i]
                if lp_[i][num] <= log_prob2t[i][num]:
                    rank_loss += (torch.sub(log_prob2t[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2a[i][num]:
                    rank_loss += (torch.sub(log_prob2a[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2v[i][num]:
                    rank_loss += (torch.sub(log_prob2v[i][num], lp_[i][num]))

            rank_lossc = 0 
            for i in range(len(log_prob)):
                num = labels_[i]
                if lp_[i][num] <= log_prob2intra[i][num]:
                    rank_lossc += (torch.sub(log_prob2intra[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2inter[i][num]:
                    rank_lossc += (torch.sub(log_prob2inter[i][num], lp_[i][num]))

            rank_loss = rank_loss + rank_lossc
            # 2. for SCL label ##########################################################
            # 0: confidence up 
            # 1: confidence drop
            Mscl_pred_t = torch.zeros_like(pred_).cuda()
            Mscl_pred_v = torch.zeros_like(pred_).cuda()
            Mscl_pred_a = torch.zeros_like(pred_).cuda()
            for index, x in enumerate(Mscl_pred_t):
                if pred_[index] != labels_[index] and pred2t_[index] == labels_[index]:
                    Mscl_pred_t[index] = 0
                if pred_[index] == labels_[index] and pred2t_[index] != labels_[index]:
                    Mscl_pred_t[index] = 1
                if pred_[index] == labels_[index] and pred2t_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2t[index][labels_[index]]:
                        Mscl_pred_t[index] = 0
                    if lp_[index][labels_[index]] > log_prob2t[index][labels_[index]]:
                        Mscl_pred_t[index] = 1

                if pred_[index] != labels_[index] and pred2a_[index] == labels_[index]:
                    Mscl_pred_a[index] = 0
                if pred_[index] == labels_[index] and pred2a_[index] != labels_[index]:
                    Mscl_pred_a[index] = 1
                if pred_[index] == labels_[index] and pred2a_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2a[index][labels_[index]]:
                        Mscl_pred_a[index] = 0
                    if lp_[index][labels_[index]] > log_prob2a[index][labels_[index]]:
                        Mscl_pred_a[index] = 1

                if pred_[index] != labels_[index] and pred2v_[index] == labels_[index]:
                    Mscl_pred_v[index] = 0
                if pred_[index] == labels_[index] and pred2v_[index] != labels_[index]:
                    Mscl_pred_v[index] = 1
                if pred_[index] == labels_[index] and pred2v_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2v[index][labels_[index]]:
                        Mscl_pred_v[index] = 0
                    if lp_[index][labels_[index]] > log_prob2v[index][labels_[index]]:
                        Mscl_pred_v[index] = 1

            Cscl_pred_intra = torch.zeros_like(pred_).cuda()
            Cscl_pred_inter = torch.zeros_like(pred_).cuda()
            for index, x in enumerate(Cscl_pred_intra):
                if pred_[index] != labels_[index] and pred2c_intra[index] == labels_[index]:
                    Cscl_pred_intra[index] = 0
                if pred_[index] == labels_[index] and pred2c_intra[index] != labels_[index]:
                    Cscl_pred_intra[index] = 1
                if pred_[index] == labels_[index] and pred2c_intra[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2intra[index][labels_[index]]:
                        Cscl_pred_intra[index] = 0
                    if lp_[index][labels_[index]] > log_prob2intra[index][labels_[index]]:
                        Cscl_pred_intra[index] = 1

                if pred_[index] != labels_[index] and pred2c_inter[index] == labels_[index]:
                    Cscl_pred_inter[index] = 0
                if pred_[index] == labels_[index] and pred2c_inter[index] != labels_[index]:
                    Cscl_pred_inter[index] = 1
                if pred_[index] == labels_[index] and pred2c_inter[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2inter[index][labels_[index]]:
                        Cscl_pred_inter[index] = 0
                    if lp_[index][labels_[index]] > log_prob2inter[index][labels_[index]]:
                        Cscl_pred_inter[index] = 1
            

            # 2. HSCL
            Mscl_tav = CKSCL_m(ft_scl, labels1=Mscl_pred_t, labels2=Mscl_pred_a, labels3=Mscl_pred_v, weight1=create_class_weight_SCL(Mscl_pred_t), weight2=create_class_weight_SCL(Mscl_pred_a), weight3=create_class_weight_SCL(Mscl_pred_v))
            Cscl_c = CKSCL_c(ft_scl, labels1=Cscl_pred_intra, labels2=Cscl_pred_inter, weight1=create_class_weight_SCL(Cscl_pred_intra), weight2=create_class_weight_SCL(Cscl_pred_inter))

        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        all_log_probs.append(log_prob)  # 保存当前批次的 log_prob
        
        if train:
            if args.calibrate:
                loss = loss + args.rank_coff * rank_loss
            if args.contrastlearning:
                loss = loss + args.mscl_coff * Mscl_tav + args.cscl_coff * Cscl_c
            loss.backward()
            optimizer.step()
            

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el, all_log_probs


def KD_model(model1, model2, args, loss_function, dataloader, epoch, cuda, modals, optimizer1=None, optimizer2=None, train=False, dataset='IEMOCAP'):
    losses, preds1, preds2, labels = [], [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer1 or optimizer2 is not None
    if train:
        model1.train()
        model2.train()
    else:
        model1.eval()
        model2.eval()

    seed_everything(args.seed)
    all_log_probs1 = []
    all_log_probs2 = []
    def distillation_loss(log_prob1, log_prob2, temperature=2):
        """计算蒸馏损失(KL散度)"""
        # 对log_prob先进行温度缩放
        log_prob1 = log_prob1 / temperature
        log_prob2 = log_prob2 / temperature
        
        # 计算软标签
        soft_targets1 = F.softmax(log_prob1, dim=1)
        soft_targets2 = F.softmax(log_prob2, dim=1)
        # print("log_prob1:", log_prob1)
        # print("soft_targets1:", soft_targets1)

        loss1 = -F.kl_div(soft_targets1, soft_targets2, reduction='batchmean')
        loss2 = -F.kl_div(soft_targets2, soft_targets1, reduction='batchmean')
        # 返回平均损失
        return loss1*0.8 + loss2*0.2
    
    for data in tqdm(dataloader):
        if train:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
        
        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
                # 1. Prepare input features for multi-modal fusio
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
                
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                # 初始化 textf，假设是将多个文本特征拼接在一起
                textf = torch.cat([textf1, textf2, textf3, textf4], dim=-1)
            else:
                raise NotImplementedError
        # 2. Forward pass through both models
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob1, e_i, e_n, e_t, e_l, hidden1 = model1(textf, qmask, umask, lengths, acouf, visuf)
            log_prob2, e_i, e_n, e_t, e_l, hidden2 = model2(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob1, e_i, e_n, e_t, e_l, hidden1 = model1([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            log_prob2, e_i, e_n, e_t, e_l, hidden2 = model2([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob1, e_i, e_n, e_t, e_l, hidden1 = model1([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            log_prob2, e_i, e_n, e_t, e_l, hidden2 = model2([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob1, e_i, e_n, e_t, e_l, hidden1 = model1(textf, qmask, umask, lengths)
            log_prob2, e_i, e_n, e_t, e_l, hidden2 = model2(textf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])

        # 3. Calculate the loss for both models
        loss1 = loss_function(log_prob1, label)
        loss2 = loss_function(log_prob2, label)
        
        # 4. Mutual Knowledge Distillation Loss (can be customized)
        # Using the output probabilities for distillation
        kd_loss = distillation_loss(log_prob1, log_prob2, temperature=2.0)
        
        loss = loss1 + loss2 + kd_loss

        if train:
            loss.backward()
            # optimizer2.step()
            # loss_2.backward()
            # optimizer1.step()

        # log_prob2 = (log_prob1*0.6685+log_prob2*0.6609)/1.3294
        
        preds1.append(torch.argmax(log_prob1, 1).cpu().numpy())
        preds2.append(torch.argmax(log_prob2, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        all_log_probs1.append(log_prob1)
        all_log_probs2.append(log_prob2)

    if preds1:
        preds1 = np.concatenate(preds1)
        preds2 = np.concatenate(preds2)
        labels = np.concatenate(labels)
        temp1 = torch.cat(all_log_probs1,dim=0)
        temp2 = torch.cat(all_log_probs2,dim=0)
        # 当 preds1 错误而 preds2 正确时，更新 preds1
        for i in range(len(preds1)):
            if preds1[i] != labels[i] and preds2[i] == labels[i]:
                # print(all_log_probs1)
                # 获取 log_prob1 中第二大的类别的索引
                if temp1[i].numel() >= 2:  # numel() 返回张量中的元素数量
                    second_class_index = torch.argsort(temp1[i])[-2].item()
                    preds1[i] = second_class_index
        
        for i in range(len(preds1)):
            if preds2[i] != labels[i] and preds1[i] == labels[i]:
                # print(all_log_probs1)
                # 获取 log_prob1 中第二大的类别的索引
                if temp2[i].numel() >= 2:  # numel() 返回张量中的元素数量
                    second_class_index = torch.argsort(temp2[i])[-2].item()
                    preds2[i] = second_class_index
        # 计算 preds1 正确而 preds2 错误的样本数
        preds1_correct_preds2_wrong = np.sum((preds1 == labels) & (preds2 != labels))
        
        # 计算 preds1 错误而 preds2 正确的样本数
        preds1_wrong_preds2_correct = np.sum((preds1 != labels) & (preds2 == labels))
        print("preds1_correct_preds2_wrong:",preds1_correct_preds2_wrong,'\n'
              "preds1_wrong_preds2_correct:",preds1_wrong_preds2_correct)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy1 = round(accuracy_score(labels, preds1)*100, 2)
    avg_fscore1 = round(f1_score(labels,preds1, average='weighted')*100, 2)
    avg_accuracy2 = round(accuracy_score(labels, preds2)*100, 2)
    avg_fscore2 = round(f1_score(labels,preds2, average='weighted')*100, 2)

    return avg_loss, labels, preds1, preds2, avg_accuracy1, avg_accuracy2, avg_fscore1, avg_fscore2, all_log_probs1, all_log_probs2