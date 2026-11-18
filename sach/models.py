"""Teacher network, batch-local graph, and modality-specific hash heads."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import OneHotEncoder


temp = 0.6


def similarity_matrix(l1, l2):
    n = len(l1)
    S = torch.randn((n, n))
    for i in range(n):
        for j in range(n):
            if l1[i] == l2[j]:
                S[i][j] = 1
            else:
                S[i][j] = 0
    return S


class TransformerEncoder(nn.Module):
    """Graph-conditioned attention block used by the archived training path."""

    def __init__(
        self, Q_dim, K_dim, V_dim, d_model=1024, dim_feedforward=2048, dropout=0.5
    ) -> None:
        super(TransformerEncoder, self).__init__()

        self.self_attn = SelfAttention(Q_dim, K_dim, V_dim, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, V_dim),
            nn.BatchNorm1d(V_dim),
            nn.Tanh(),
        )

        self.activation = nn.Tanh()

    def forward(self, src, anchor_1, anchor_2, label_graph):
        src2, _ = self.self_attn(src, anchor_1, anchor_2, label_graph)
        src = src2

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = self.decoder(src)
        return src


class SelfAttention(nn.Module):
    def __init__(self, Q_dim, K_dim, V_dim, d_model) -> None:
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.Q_layer = nn.Linear(Q_dim, d_model)
        self.K_layer = nn.Linear(K_dim, d_model)
        self.V_layer = nn.Linear(V_dim, d_model)

    def forward(self, Q, K, V, label_graph):
        Q = self.Q_layer(Q)
        # Keep the original shared query/key projection and parameter layout.
        K = self.Q_layer(K)
        V = self.V_layer(V)
        attention_score = (Q @ K.T) / math.sqrt(self.d_model)
        if label_graph != None:
            attention_score = label_graph.mul(attention_score)
        attention_prob = F.softmax(attention_score, dim=-1)
        context = attention_prob @ V

        return context, attention_prob


class MyNet(nn.Module):
    """Generate joint training codes from paired image and text features."""

    def __init__(self, code_len, ori_featI, ori_featT, num):
        super(MyNet, self).__init__()
        self.code_len = code_len

        # Retain construction order and state keys for the dormant attention blocks.
        self.img_transformer = TransformerEncoder(Q_dim=ori_featI, K_dim=ori_featI, V_dim=ori_featI)
        self.txt_transformer = TransformerEncoder(Q_dim=ori_featT, K_dim=ori_featT, V_dim=ori_featT)

        self.sc = SpectralClustering(n_clusters=num, affinity="nearest_neighbors")

        self.encoderIMG = nn.Sequential(
            nn.Linear(ori_featI, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.gcnI1 = nn.Linear(512, 512)
        self.BNI1 = nn.BatchNorm1d(512)
        self.actI1 = nn.ReLU(inplace=True)
        self.S_BNI1 = nn.BatchNorm1d(512)
        self.decoderIMG = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ori_featI),
            nn.BatchNorm1d(ori_featI),
        )

        self.encoderTXT = nn.Sequential(
            nn.Linear(ori_featT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.gcnT1 = nn.Linear(512, 512)
        self.BNT1 = nn.BatchNorm1d(512)
        self.actT1 = nn.ReLU(inplace=True)

        self.decoderTXT = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ori_featT),
            nn.BatchNorm1d(ori_featT),
        )

        self.gcnJ1 = nn.Linear(512, 512)
        self.BNJ1 = nn.BatchNorm1d(512)
        self.actJ1 = nn.ReLU(inplace=True)

        self.HJ = nn.Linear(512, code_len)

        self.img_fc = nn.Linear(2 * 512, code_len)
        self.txt_fc = nn.Linear(2 * 512, code_len)
        self.HC = nn.Linear(3 * code_len, code_len)

        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)
        self.HJBN = nn.BatchNorm1d(code_len)
        self.HBN = nn.BatchNorm1d(code_len)

    def forward(self, XI, XT, affinity_A):
        self.batch_num = XI.size(0)

        VI = self.encoderIMG(XI)
        VT = self.encoderTXT(XT)

        VI = F.normalize(VI, dim=1)
        VT = F.normalize(VT, dim=1)

        # Cluster detached text and image rows in the original fit order.
        vt = VT.cpu().detach().numpy()
        vi = VI.cpu().detach().numpy()
        prediction1 = self.sc.fit_predict(vt)
        prediction2 = self.sc.fit_predict(vi)
        enc1 = OneHotEncoder()
        enc1.fit(prediction1.reshape(-1, 1))
        one_hot1 = enc1.transform(prediction1.reshape(-1, 1)).toarray()
        enc2 = OneHotEncoder()
        enc2.fit(prediction1.reshape(-1, 1))
        # Preserve the baseline's reuse of the text-fitted encoder for image IDs.
        one_hot2 = enc1.transform(prediction2.reshape(-1, 1)).toarray()
        one_hot1 = torch.from_numpy(one_hot1)
        one_hot2 = torch.from_numpy(one_hot2)
        S1 = one_hot1.mm(one_hot1.t())
        S2 = one_hot2.mm(one_hot2.t())
        S1 = S1.to("cuda")
        S2 = S2.to("cuda")
        affinity_A = affinity_A + 0.7 * S1 + 0.3 * S2
        affinity_A = affinity_A.to(torch.float32)

        VgcnI = self.gcnI1(VI)
        VgcnI = affinity_A.mm(VgcnI)
        VgcnI = self.BNI1(VgcnI)
        VgcnI = self.actI1(VgcnI)

        VgcnT = self.gcnT1(VT)
        VgcnT = affinity_A.mm(VgcnT)
        VgcnT = self.BNT1(VgcnT)
        VgcnT = self.actT1(VgcnT)

        # Cross-modal integration links each paired image and text with an identity edge.
        VC = torch.cat((VI, VT), 0)
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        S_cma = torch.cat(
            (torch.cat((affinity_A, II), 1), torch.cat((II, affinity_A), 1)),
            0,
        )

        VJ1 = self.gcnJ1(VC)
        VJ1 = S_cma.mm(VJ1)
        VJ1 = self.BNJ1(VJ1)
        VJ1 = VJ1[: self.batch_num, :] + VJ1[self.batch_num :, :]
        VJ = self.actJ1(VJ1)

        HJ = self.HJ(VJ)
        HJ = self.HJBN(HJ)

        HI = self.HIBN(self.img_fc(torch.cat((VgcnI, VJ), 1)))
        HT = self.HTBN(self.txt_fc(torch.cat((VJ, VgcnT), 1)))

        H = torch.tanh(self.HBN(self.HC(torch.cat((HI, HJ, HT), 1))))

        B = torch.sign(H)

        DeI_feat = self.decoderIMG(H + torch.tanh(HI))
        DeT_feat = self.decoderTXT(H + torch.tanh(HT))

        return HI, HT, H, B, DeI_feat, DeT_feat, VgcnI, VgcnT


class Img_Net(nn.Module):
    """Map image features to relaxed hash codes."""

    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()

        self.fc1 = nn.Linear(img_feat_len, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)

        return HI


class Txt_Net(nn.Module):
    """Map text features to relaxed hash codes."""

    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()

        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)

        return HT
