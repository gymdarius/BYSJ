import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)

        try:
            # 检查标签和预测的类别数
            test_lbls_np = test_lbls.detach().cpu().numpy()
            best_proba_np = best_proba.detach().cpu().numpy()
            
            # 确保分类器的输出列数与实际标签的类别数一致
            actual_classes = test_lbls_np.shape[1] if len(test_lbls_np.shape) > 1 else len(np.unique(test_lbls_np))
            pred_classes = best_proba_np.shape[1]
            
            if actual_classes != pred_classes:
                print(f"警告: 标签类别数({actual_classes})与预测类别数({pred_classes})不匹配")
        
                if actual_classes < pred_classes:
                    # 截取预测结果并重新归一化
                    print(f"截取预测结果从{pred_classes}列到{actual_classes}列")
                    best_proba_np = best_proba_np[:, :actual_classes]
                    # 重新归一化，确保每行和为1
                    row_sums = best_proba_np.sum(axis=1, keepdims=True)
                    best_proba_np = best_proba_np / row_sums
                else:
                    print("无法计算AUC，类别数不匹配且无法调整")
                    auc_score_list.append(0.0)
                    continue
                    
            auc_score_list.append(roc_auc_score(y_true=test_lbls_np,
                                                y_score=best_proba_np,
                                                multi_class='ovr'
                                                ))
        except ValueError as e:
            print(f"计算AUC时出错: {e}")
            print(f"预测形状: {best_proba.shape}, 标签形状: {test_lbls.shape}")
            # 跳过AUC计算，添加默认值
            auc_score_list.append(0.0)


    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()
