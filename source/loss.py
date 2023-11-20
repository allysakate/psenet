import torch
import torch.nn.functional as F


def gradient(pred):
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def smooth_loss(pred_map):
    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss = dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
    return loss


class TVLoss(torch.nn.Module):
    def forward(self, x):
        x = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        return torch.mean(h_tv) + torch.mean(w_tv)


class Loss:
    def __init__(self, lambda_tv, lambda_cc, lambda_recon, lambda_smooth) -> None:
        self.lambda_tv = lambda_tv
        self.lambda_cc = lambda_cc
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        self.mse = torch.nn.MSELoss()
        self.tv = TVLoss()

    def get_loss(self, pred_im, pred_gamma, pseudo_gt, syn_pseudo):
        recon_loss = F.l1_loss(pred_im, pseudo_gt)
        dle_pred_cc = torch.mean(pred_im, dim=1, keepdims=True)
        cc_loss = (
            F.l1_loss(pred_im[:, 0:1, :, :], dle_pred_cc)
            + F.l1_loss(pred_im[:, 1:2, :, :], dle_pred_cc)
            + F.l1_loss(pred_im[:, 2:3, :, :], dle_pred_cc)
        ) * (1 / 3)

        le_smooth_loss = smooth_loss(pred_gamma)
        tv_loss = self.tv(pred_gamma)

        try:
            l1_loss = F.l1_loss(pred_im, syn_pseudo)
            loss = self.lambda_recon * recon_loss
            loss += self.lambda_recon * l1_loss
        except Exception:
            loss = self.lambda_recon * recon_loss
        loss += self.lambda_cc * cc_loss
        loss += self.lambda_smooth * le_smooth_loss
        loss += self.lambda_tv * tv_loss

        return loss
