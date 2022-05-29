import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_base import AUXBase
from common import utils
from module.rl_module import CFPredictor


class CRESP(AUXBase):

    def __init__(self, action_shape, extr_latent_dim, nstep_of_rsd, hidden_dim,
                 output_dim, act_seq_out_dim, omg_seq_out_dim, l, rs_fc, extr_lr,
                 extr_beta, omega_opt_mode=None, num_sample=256, discount_of_rs=0.8,
                 temperature=0.1, opt_mode='min', opt_num=5, device='cpu', **kwargs):
        super().__init__()
        action_dim = action_shape[0]
        act_seq_in_dim = nstep_of_rsd*action_dim
        # Initialize hyperparameters
        self.nstep_of_rsd = nstep_of_rsd
        self.rs_fc = rs_fc
        self.discount_of_rs = discount_of_rs
        self.pred_temp = temperature
        self.output_dim = output_dim
        self.opt_mode = opt_mode
        self.opt_num = opt_num
        self.device = device

        # Initialize modules
        self.network = CFPredictor(extr_latent_dim,
                                   act_seq_in_dim,
                                   nstep_of_rsd,
                                   hidden_dim,
                                    act_seq_out_dim,
                                   omg_seq_out_dim,
                                   output_dim, l,
                                   rs_fc=rs_fc,
                                   omega_opt_mode=omega_opt_mode,
                                   num_sample=num_sample).to(device)

        # Initialize optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=extr_lr, betas=(extr_beta, 0.999))

    def _prepare_data(self, data, num_aug):
        with torch.no_grad():
            traj_a, traj_r = data['traj_a'], data['traj_r']
            batch_size = traj_r.size(0)
            a_seq = traj_a.repeat(num_aug, 1, 1) # (batch_size*num_aug, rs_dim, a_dim)
            discount = (self.discount_of_rs ** torch.arange(
                self.nstep_of_rsd).to(traj_r.device)).unsqueeze(0)
            traj_r *= discount
            r_seq = traj_r.repeat(num_aug, 1) # (batch_size*num_aug, rs_dim)
        return a_seq, r_seq, batch_size

    def calc_psi(self, r_seq, w_seq):
        if self.rs_fc:
            r_seq = self.network.forward_rs(r_seq)
        inner_product = w_seq @ r_seq.t()
        psi_targ_cos = np.pi / 2 * torch.cos(inner_product) # (num_sample, batch_size*num_aug)
        psi_targ_sin = np.pi / 2 * torch.sin(inner_product) # (num_sample, batch_size*num_aug)
        return torch.stack([psi_targ_cos, psi_targ_sin], dim=-1) # (num_sample, batch_size*num_aug, 2)


    def update_extr(self, data, aug_s, num_aug):
        a_seq, r_seq, batch_size = self._prepare_data(data, num_aug)
        labels = torch.arange(batch_size).long().to(r_seq.device)

        w_seq = self.network.omega
        psi_targ = self.calc_psi(r_seq, w_seq)  # (num_sample, batch_size*num_aug, 2)
        
        psi_cos, psi_sin = self.network(aug_s, a_seq, w_seq).chunk(2, -1) # (num_sample, batch_size*num_aug, output_dim)
        psi = torch.stack([psi_cos, psi_sin], dim=0).transpose(0, -1) # (output_dim, num_sample, batch_size*num_aug, 2)
        # if psi.ndim == 3:
        #     psi = psi.unsqueeze(0)

        # MSE
        psi_error = (psi - psi_targ.unsqueeze(0)).pow(2)
        if self.output_dim != 1:
            psi_error = utils.rank(psi_error,
                                   psi_error.mean([-1, -2, -3]).detach(),
                                   dim=0,
                                   num=self.opt_num,
                                   mode=self.opt_mode) # (opt_num, num_sample, batch_size*num_aug, 2)

        # idx = np.random.randint(0, psi_error.size(0), psi_error.size(2))
        # loss_psi_mse = psi_error[idx, :, np.arange(idx.shape[0])].mean(0).sum()
        # loss_psi_std = torch.max(torch.norm(psi_error - psi_error.mean(0), dim=1).view(
        #     psi_error.size(0), -1, num_aug, 2
        # ), dim=0)[0].sum(-1).mean()
        loss_psi_mse = psi_error.sum(0).sum(-1).mean() * num_aug
        loss_psi_std = psi_error.std(1).mean() * num_aug

        # Cross Entropy
        psi_cl_targ = psi_targ.transpose(0, 1).reshape(psi_targ.size(1), -1).view(
            batch_size, num_aug, -1) # (batch_size, num_aug, num_sample*2)
        psi_cl = psi.transpose(1, 2).reshape(psi.size(0), psi.size(2), -1).view(
            psi.size(0), batch_size, num_aug, -1) # (output_dim, batch_size, num_aug, num_sample*2)

        loss_psi_cl, acc = utils.compute_cl_loss(psi_cl[:,:,0], psi_cl_targ[:,0], labels, None, self.pred_temp, True)
        # loss_psi_cl2, acc2 = utils.compute_cl_loss(psi_cl[:,:,1], psi_cl_targ[:,1], labels, None, self.pred_temp, True)
        if self.output_dim != 1:
            loss_psi_cl = utils.rank(loss_psi_cl,
                                      loss_psi_cl.detach(),
                                      dim=0,
                                      num=self.opt_num,
                                      mode=self.opt_mode) # (opt_num,)
            # loss_psi_cl2 = utils.rank(loss_psi_cl2,
            #                           loss_psi_cl2.detach(),
            #                           dim=0,
            #                           num=self.opt_num,
            #                           mode=self.opt_mode) # (opt_num,)
        # loss_psi_cl = loss_psi_cl1 + loss_psi_cl2

        # loss_psi = loss_psi_mse.sum() + 0.01 * loss_psi_std + loss_psi_cl.sum()
        loss_psi = loss_psi_mse + loss_psi_cl.sum()
        opt_dict = dict(opt_p=self.optimizer)
        info_dict = dict(LossPsi=loss_psi_cl.clone(), LossPsiMSE=loss_psi_mse.clone(),
                         LossPsiCL=loss_psi_cl.clone(), PsiCLAcc=acc,
                         PsiMSESTD=loss_psi_std.clone())
        return loss_psi, opt_dict, info_dict


    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('LossPsi', average_only=True)
        logger.log_tabular('LossPsiMSE', average_only=True)
        logger.log_tabular('PsiMSESTD', average_only=True)
        logger.log_tabular('LossPsiCL', average_only=True)
        logger.log_tabular('PsiCLAcc', average_only=True)
