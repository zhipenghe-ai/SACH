"""Training controller for the SACH release."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .checkpoint import create_checkpoint_path, load_hash_heads, save_hash_heads
from .data import get_loader_coco, get_loader_flickr, get_loader_nus
from .evaluation import calculate_top_map, compress, p_topK, pr_curve
from .models import Img_Net, MyNet, TransformerEncoder, Txt_Net

import lightly.loss as MYloss


def _validated_sample_ids(record_index, dataset_size):
    """Read original dataset IDs in the order of the collected teacher codes."""
    sample_ids = []
    for value in record_index:
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Expected a scalar integer sample ID, got {value!r}")
        if not 0 <= value < dataset_size:
            raise IndexError(f"Sample ID {value} is outside [0, {dataset_size})")
        sample_ids.append(int(value))
    return sample_ids


class PCIRH:
    """Coordinate code generation and modality hash-function learning."""

    def __init__(self, log, config):
        self.config = config
        self.log = log

        data_root = getattr(config, "DATA_ROOT", "./datasets")
        loaders = {
            "MIRFlickr": get_loader_flickr,
            "NUSWIDE": get_loader_nus,
            "COCO": get_loader_coco,
        }
        if config.DATASET not in loaders:
            raise ValueError(
                f"Unknown dataset {config.DATASET!r}. Choose MIRFlickr, NUSWIDE or COCO."
            )
        dataloader, data_train = loaders[config.DATASET](config.BATCH_SIZE, data_root)

        self.I_tr, self.T_tr, self.L_tr = data_train

        self.train_num = self.I_tr.shape[0]
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]

        self.num = self.L_tr.shape[1]

        self.train_loader = dataloader["train"]
        self.test_loader = dataloader["query"]
        self.database_loader = dataloader["database"]

        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        self.S_cmm, self.S_I, self.S_T = self.cal_similarity(img_norm, txt_norm)

        self.mynet = MyNet(
            code_len=self.config.HASH_BIT,
            ori_featI=self.img_feat_len,
            ori_featT=self.txt_feat_len,
            num=self.config.num,
        ).cuda()
        self.imgnet = Img_Net(
            code_len=self.config.HASH_BIT,
            img_feat_len=self.img_feat_len,
        ).cuda()
        self.txtnet = Txt_Net(code_len=self.config.HASH_BIT, txt_feat_len=self.txt_feat_len).cuda()

        self.opt_mynet = torch.optim.Adam(self.mynet.parameters(), lr=config.LR_MyNet)
        self.opt_imgnet = torch.optim.Adam(self.imgnet.parameters(), lr=config.LR_IMG)
        self.opt_txtnet = torch.optim.Adam(self.txtnet.parameters(), lr=config.LR_TXT)

        self.record_Lsmodel = []
        self.record_Lshfunc = []
        # Preserve anchor sampling and module initialization from the archived path.
        self.anchor_idx = np.random.permutation(self.train_num)[: self.config.anchor_nums]

        self.img_anchor = torch.Tensor(self.I_tr[self.anchor_idx, :]).cuda()
        self.txt_anchor = torch.Tensor(self.T_tr[self.anchor_idx, :]).cuda()

        self.img_TEs_enc = TransformerEncoder(
            Q_dim=self.txt_feat_len, K_dim=self.txt_feat_len, V_dim=self.img_feat_len
        ).cuda()
        self.txt_TEs_enc = TransformerEncoder(
            Q_dim=self.img_feat_len, K_dim=self.img_feat_len, V_dim=self.txt_feat_len
        ).cuda()

        self.TEs_optimizer = torch.optim.Adam(
            [
                {"params": self.img_TEs_enc.parameters(), "lr": self.config.lr},
                {"params": self.txt_TEs_enc.parameters(), "lr": self.config.lr},
            ]
        )

    def train_method(self, epoch):
        coll_B = list([])
        record_index = list([])
        Ls_method = 0

        self.mynet.train()
        self.img_TEs_enc.train()
        self.txt_TEs_enc.train()
        for No, (img, txt, _, index_) in enumerate(self.train_loader):

            img = Variable(img.cuda().to(torch.float))
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

            S_batch = self.S_cmm[index_, :][:, index_].cuda()

            Iind1, Iind2 = self.NN_select(S_batch)
            Tind1, Tind2 = self.NN_select(S_batch)
            imgNN1, imgNN2 = img[Iind1, :], img[Iind2, :]
            txtNN1, txtNN2 = txt[Tind1, :], txt[Tind2, :]

            imgN = 0.7 * img + 0.2 * imgNN2 + 0.1 * imgNN1
            txtN = 0.7 * txt + 0.2 * txtNN2 + 0.1 * txtNN1

            HI, HT, H, B, DeI_feat, DeT_feat, VgcnI, VgcnT = self.mynet(imgN, txtN, S_batch)

            coll_B.extend(B.cpu().data.numpy())
            record_index.extend(index_)

            self.opt_mynet.zero_grad()
            self.TEs_optimizer.zero_grad()

            loss_CCR = self.config.lambda1 * (
                F.mse_loss(DeI_feat, imgN) + F.mse_loss(DeT_feat, txtN)
            )
            loss_2 = self.loss_method(HI, HT, H, B, S_batch)

            # Decoupled contrastive loss on the two graph outputs.
            DLCloss = MYloss.DCLLoss(self.config.temperature)
            DLC = DLCloss(VgcnI, VgcnT)

            # The attention reconstruction terms below are logged separately.
            g1 = GEN_S_GPU(txt, txt)
            g2 = GEN_S_GPU(img, img)

            m, n1 = g1.sort()
            g1[
                torch.arange(txt.shape[0]).view(-1, 1).repeat(1, self.config.top_size).view(-1),
                n1[:, : self.config.top_size].contiguous().view(-1),
            ] = 0.0

            m, n1 = g2.sort()
            g2[
                torch.arange(img.shape[0]).view(-1, 1).repeat(1, self.config.top_size).view(-1),
                n1[:, : self.config.top_size].contiguous().view(-1),
            ] = 0.0

            img_neighbour = self.img_TEs_enc(txt, txt, img, g1)
            txt_neighbour = self.txt_TEs_enc(img, img, txt, g2)
            bt1 = self.config.bt1
            bt2 = 5
            myloss1 = bt1 * F.mse_loss(img_neighbour, img)
            myloss2 = bt2 * F.mse_loss(txt_neighbour, txt)

            loss_model = loss_CCR + loss_2 + DLC
            Ls_method = (Ls_method + loss_model).item()

            loss_model.backward()
            self.opt_mynet.step()
            self.TEs_optimizer.step()

            if (No + 1) % (
                self.train_num // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL
            ) == 0:
                self.log.info(
                    "Epoch [%d/%d], Iter [%d/%d] Ls1=%.4f, Ls2=%.4f,myloss1=%.4f,myloss2=%.4f,Ls_model: %.4f"
                    % (
                        epoch + 1,
                        self.config.NUM_EPOCH,
                        No + 1,
                        self.train_num // self.config.BATCH_SIZE,
                        loss_CCR,
                        loss_2,
                        bt1 * myloss1,
                        bt2 * myloss2,
                        loss_model.item(),
                    )
                )

        coll_B = np.array(coll_B)

        self.record_Lsmodel.append(Ls_method)

        return coll_B, record_index

    def train_Hashfunc(self, coll_B, record_index, epoch):
        sample_ids = _validated_sample_ids(record_index, self.I_tr.shape[0])
        if not sample_ids or len(sample_ids) != len(coll_B):
            raise ValueError("Provide one sample ID per teacher code and at least one code")

        self.imgnet.train()
        self.txtnet.train()

        Ls_hfunc = 0

        B = torch.from_numpy(coll_B).cuda()

        img = torch.Tensor(self.I_tr[sample_ids, :]).cuda()
        txt = torch.Tensor(self.T_tr[sample_ids, :]).cuda()

        img_norm = F.normalize(img)
        txt_norm = F.normalize(txt)

        # Features and codes share the collected order, but S_cmm uses original dataset IDs.
        graph_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=self.S_cmm.device)
        batch_size = self.config.BATCH_SIZE
        for start in range(0, len(sample_ids), batch_size):
            batch = slice(start, start + batch_size)
            img_batch = img_norm[batch, :]
            txt_batch = txt_norm[batch, :]
            B_batch = B[batch, :]
            batch_ids = graph_ids[batch]
            S_batch = self.S_cmm[batch_ids[:, None], batch_ids[None, :]]

            hfunc_BI = self.imgnet(img_batch)
            hfunc_BT = self.txtnet(txt_batch)

            self.opt_imgnet.zero_grad()
            self.opt_txtnet.zero_grad()

            loss_f1 = (
                F.mse_loss(hfunc_BI, B_batch)
                + F.mse_loss(hfunc_BT, B_batch)
                + F.mse_loss(hfunc_BI, hfunc_BT)
            )

            S_BI_BT = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BT).t())
            S_BI_BI = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BI).t())
            S_BT_BT = F.normalize(hfunc_BT).mm(F.normalize(hfunc_BT).t())
            loss_f2 = (
                F.mse_loss(S_BI_BT, S_batch)
                + F.mse_loss(S_BI_BI, S_batch)
                + F.mse_loss(S_BT_BT, S_batch)
            )

            loss_hfunc = loss_f1 + self.config.beta * loss_f2
            Ls_hfunc = (Ls_hfunc + loss_hfunc).item()

            loss_hfunc.backward()

            self.opt_imgnet.step()
            self.opt_txtnet.step()

        self.log.info(
            "Epoch [%d/%d], Ls_hfunc: %.4f" % (epoch + 1, self.config.NUM_EPOCH, loss_hfunc.item())
        )

        self.record_Lshfunc.append(Ls_hfunc)

        return self.imgnet, self.txtnet

    def performance_eval(self):

        self.log.info("--------------------Evaluation: MAP (full gallery)-------------------")
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(
            self.database_loader, self.test_loader, self.imgnet, self.txtnet
        )

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        I_T = p_topK(qB=qu_BI, rB=re_BT, query_label=qu_L, retrieval_label=re_L)
        T_I = p_topK(qB=qu_BT, rB=re_BI, query_label=qu_L, retrieval_label=re_L)
        p1, r1 = pr_curve(qB=qu_BI, rB=re_BT, query_label=qu_L, retrieval_label=re_L)
        p2, r2 = pr_curve(qB=qu_BT, rB=re_BI, query_label=qu_L, retrieval_label=re_L)

        return MAP_I2T, MAP_T2I, I_T, T_I, p1, r1, p2, r2

    def eval2(self):

        self.log.info("--------------------Evaluation: MAP (full gallery)-------------------")
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(
            self.database_loader, self.test_loader, self.imgnet, self.txtnet
        )

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I

    def NN_select(self, S):

        m, n1 = S.sort()
        ind1 = n1[:, -4:-1][:, 0]
        ind2 = n1[:, -4:-1][:, 1]

        return ind1, ind2

    def cal_similarity(self, F_I, F_T):
        batch_size = F_I.size(0)
        size = batch_size
        top_size = self.config.K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = self.config.a1 * S_I + (1 - self.config.a1) * S_T

        m, n1 = S1.sort()
        S1[
            torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1),
            n1[:, :top_size].contiguous().view(-1),
        ] = 0.0

        S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        S2 = (S2 + S2.t()) / 2
        S = self.config.a2 * S1 + (1 - self.config.a2) * S2

        return S, S_I, S_T

    def loss_method(self, HI, HT, H, B, S_batch):

        HI_norm = F.normalize(HI)
        HT_norm = F.normalize(HT)
        H_norm = F.normalize(H)

        HI_HI = HI_norm.mm(HI_norm.t())
        HT_HT = HT_norm.mm(HT_norm.t())
        H_H = H_norm.mm(H_norm.t())

        # loss_ACR = self.config.lambda2 * (
        #         F.mse_loss(S_batch, HI_HI) + F.mse_loss(S_batch, HT_HT) + F.mse_loss(S_batch, H_H))
        loss_ACR = self.config.lambda2 * (F.mse_loss(S_batch, HI_HI) + F.mse_loss(S_batch, HT_HT))
        loss_DIS = F.mse_loss(H, B)

        return loss_ACR + loss_DIS

    def save_checkpoints(self):
        if getattr(self, "_checkpoint_path", None) is None:
            filename = f"{self.config.DATASET}_{self.config.HASH_BIT}bits.pth"
            self._checkpoint_path = create_checkpoint_path(self.config.MODEL_DIR, filename)
        path = save_hash_heads(self._checkpoint_path, self.imgnet, self.txtnet)
        self.log.info("Saved hash-head checkpoint to %s", path)
        return path

    def load_checkpoints(self, file_name):
        path = Path(self.config.MODEL_DIR) / file_name
        path = load_hash_heads(path, self.imgnet, self.txtnet)
        self.log.info("Loaded hash-head checkpoint from %s", path)
        return path


def GEN_S_GPU(label_1, label_2):
    aff = torch.matmul(label_1, label_2.T)
    affinity_matrix = aff.float()
    # Map the dot products through a sigmoid before rescaling to a signed range.
    affinity_matrix = 1 / (1 + torch.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix
