import torch
import torch.nn.functional as F

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math






def train_sp_norm(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        # rationales, logits = model(inputs, masks)
        rationales, logits = model.perturb_forward(inputs, masks,args.perturb_rate)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr



        loss.backward()


        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    print('---------------------------train_sp={}----------------'.format(np.mean(train_sp)))

    return precision, recall, f1_score, accuracy


def train_mochang(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    """
    由perturbation版本的train_sp_norm改造而来， 将选择的rationale的模长作为一个额外的损失函数
    """

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        # rationales, logits = model(inputs, masks)
        rationales, logits, mochang = model.mochang_forward(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        if args.mochang_loss=='log':
            mochang_loss=-args.mochang_lambda*torch.mean(torch.log(1+mochang))
        elif args.mochang_loss=='daoshu':
            mochang_loss = +args.mochang_lambda * torch.mean(1/(1+mochang))
        else:
            mochang_loss = -args.mochang_lambda * torch.mean(mochang)

        loss = cls_loss + sparsity_loss + continuity_loss+mochang_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr



        loss.backward()


        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    print('---------------------------train_sp={}----------------'.format(np.mean(train_sp)))

    return precision, recall, f1_score, accuracy


def train_pred_mochang(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    """
    由perturbation版本的train_sp_norm改造而来， 将选择的rationale的模长作为一个额外的损失函数,使用predictor的encoder作为衡量模长的工具
    """

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        # rationales, logits = model(inputs, masks)
        rationales, logits= model.forward(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])



        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr



        loss.backward()


        optimizer.step()

        optimizer.zero_grad()

        #模长只更新generator
        name1 = []
        name2 = []

        for idx,p in model.cls.named_parameters():
            if p.requires_grad==True:
                name1.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False

        rationales, logits, last_hid_state = model.forward_predemb(inputs, masks)
        mochang=torch.norm(last_hid_state, p=2, dim=1)

        if args.mochang_loss=='log':
            if args.add==1:
                mochang_loss=-args.mochang_lambda*torch.mean(torch.log(1+mochang))
            else:
                mochang_loss = -args.mochang_lambda * torch.mean(torch.log(1e-6 + mochang))
        elif args.mochang_loss=='daoshu':
            if args.add == 1:
                mochang_loss = +args.mochang_lambda * torch.mean(1/(1+mochang))
            else:
                mochang_loss = +args.mochang_lambda * torch.mean(1 / (1e-6 + mochang))
        else:
            mochang_loss = -args.mochang_lambda * torch.mean(mochang)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss =sparsity_loss + continuity_loss +mochang_loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        n1 = 0
        n2 = 0
        for idx, p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad = True
                n1 += 1
        for idx, p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1



        #计算训练准确率
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    print('---------------------------train_sp={}----------------'.format(np.mean(train_sp)))

    return precision, recall, f1_score, accuracy


def train_pred_mochang_withoutacc(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    """
    由perturbation版本的train_sp_norm改造而来， 将选择的rationale的模长作为一个generator的损失函数,使用predictor的encoder作为衡量模长的工具,acc仅用于更新predictor
    """

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)



        # rationales, cls_logits
        # rationales, logits = model(inputs, masks)
        rationales=model.get_rationale(inputs, masks)
        logits=model.pred_with_rationale(inputs, masks,torch.detach(rationales))

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])



        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr






        loss.backward()


        optimizer.step()

        optimizer.zero_grad()

        #模长只更新generator，冻住predictor的参数






        name1 = []
        name2 = []

        for idx,p in model.cls.named_parameters():
            if p.requires_grad==True:
                name1.append(idx)
                p.requires_grad=False
        for idx,p in model.cls_fc.named_parameters():
            if p.requires_grad == True:
                name2.append(idx)
                p.requires_grad = False

        rationales, logits, last_hid_state = model.forward_predemb(inputs, masks)
        mochang=torch.norm(last_hid_state, p=2, dim=1)

        if args.mochang_loss=='log':
            if args.add==1:
                mochang_loss=-args.mochang_lambda*torch.mean(torch.log(1+mochang))
            else:
                mochang_loss = -args.mochang_lambda * torch.mean(torch.log(1e-6 + mochang))
        elif args.mochang_loss=='daoshu':
            if args.add == 1:
                mochang_loss = +args.mochang_lambda * torch.mean(1/(1+mochang))
            else:
                mochang_loss = +args.mochang_lambda * torch.mean(1 / (1e-6 + mochang))
        else:
            mochang_loss = -args.mochang_lambda * torch.mean(mochang)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss =sparsity_loss + continuity_loss +mochang_loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        n1 = 0
        n2 = 0
        for idx, p in model.cls.named_parameters():
            if idx in name1:
                p.requires_grad = True
                n1 += 1
        for idx, p in model.cls_fc.named_parameters():
            if idx in name2:
                p.requires_grad = True
                n2 += 1



        #计算训练准确率
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    # grad_max, grad_mean = get_grad(model, dataset, 2, 1, device)  # 获取训练过程中的lipschitz常数
    # writer_epoch[0].add_scalar('max_grad', grad_max, writer_epoch[1])
    # writer_epoch[0].add_scalar('avg_grad', grad_mean, writer_epoch[1])
    print('---------------------------train_sp={}----------------'.format(np.mean(train_sp)))

    return precision, recall, f1_score, accuracy



def classfy(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    mochang=0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        logits, clsoutput = model.get_representation(inputs, masks)

        # computer loss
        cls_loss =F.cross_entropy(logits, labels)


        loss = cls_loss

        # update gradient
        loss.backward()

        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        mochang += torch.sum(torch.norm(clsoutput, p=2, dim=1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    mochang= mochang/ (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy,mochang



def classfy_trivial(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    mochang=0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        trivial_input=torch.randint_like(inputs,1,args.vocab_size,device=device)
        pos_num=int(trivial_input.shape[0]/2)
        trivial_input[:pos_num,0]=2
        trivial_input[pos_num:,0]=3
        labels[:pos_num]=1
        labels[pos_num:]=0


        # rationales, cls_logits
        logits,clsoutput = model.get_representation(trivial_input, masks)

        # computer loss
        cls_loss =F.cross_entropy(logits, labels)


        loss = cls_loss

        # update gradient
        loss.backward()

        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        mochang += torch.sum(torch.norm(clsoutput, p=2, dim=1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    mochang=mochang/ (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy,mochang

def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def get_grad(model,dataloader,p,use_rat,device):            #获取训练过程中的lipschitz常数
    data=0
    # device=model.device()
    model.train()
    grad=[]
    for batch,d in enumerate(dataloader):
        data=d
        inputs, masks, labels = data
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationale,logit,embedding2,cls_embed=model.grad(inputs, masks)
        loss=torch.mean(torch.softmax(logit,dim=-1)[:,1])
        cls_embed.retain_grad()
        loss.backward()
        if use_rat==0:
            k_mask=masks
        elif use_rat==1:
            k_mask=rationale[:,:,1]
        masked_grad=cls_embed.grad*k_mask.unsqueeze(-1)
        gradtemp=torch.sum(abs(masked_grad),dim=1)       #bat*256*100→bat*100,在句子长度方向相加
        gradtemp=gradtemp/torch.sum(k_mask,dim=-1).unsqueeze(-1)      #在句子长度方向取平均
        # gradtempmask=gradtemp*rationale[:,:,1]
        # gradtempmaskmean =torch.sum(gradtempmask,dim=-1)/torch.sum(rationale[:,:,1],dim=-1)    #在句子长度方向取平均
        gradtempmask = gradtemp
        norm_grad=torch.linalg.norm(gradtempmask, ord=p, dim=1)           #在维度上取norm
        # gradtempmaskmean = torch.sum(gradtempmask, dim=-1) / torch.sum(masks, dim=-1)  # 在句子长度方向取平均
        grad.append(norm_grad.clone().detach().cpu())
    grad=torch.cat(grad,dim=0)
    tem=[]
    for g in grad:
        if math.isnan(g.item()):
            continue
        else:
            tem.append(g)

    tem=torch.tensor(tem)
    maxg=torch.max(tem)*1000
    meang=torch.mean(tem)*1000
    return maxg,meang




