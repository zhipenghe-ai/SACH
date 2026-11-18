"""Encode modality features and compute the original SACH retrieval metrics."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def pr_curve(qB, rB, query_label, retrieval_label):
    qB = torch.from_numpy(qB).float()
    rB = torch.from_numpy(rB).float()
    query_label = torch.from_numpy(query_label).long()
    retrieval_label = torch.from_numpy(retrieval_label).long()
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def p_topK(qB, rB, query_label, retrieval_label, K=None):
    # Preserve the original fixed cutoffs, including the treatment of a supplied K.
    K = [1, 300, 600, 900, 1200, 1500]
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def p_topK2(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def compress(database_loader, test_loader, model_I, model_T, device=None):
    """Encode both splits on the requested device or the trainer's current CUDA device."""

    def move(tensor):
        return tensor.cuda() if device is None else tensor.to(device)

    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            var_data_I = Variable(F.normalize(move(data_I)))
            code_I = model_I(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        with torch.no_grad():
            var_data_T = Variable(F.normalize(move(torch.FloatTensor(data_T.numpy()))))
            code_T = model_T(var_data_T.to(torch.float))
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        with torch.no_grad():
            var_data_I = Variable(F.normalize(move(data_I)))
            code_I = model_I(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        with torch.no_grad():
            var_data_T = Variable(F.normalize(move(torch.FloatTensor(data_T.numpy()))))
            code_T = model_T(var_data_T.to(torch.float))
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    # The historical evaluator always uses the full gallery.
    topk = re_L.shape[0]
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
