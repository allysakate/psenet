import os
import cv2
import numpy as np
import piq
import torch
import torchvision
import torchvision.transforms as transforms
from iqa import IQA
from loss import Loss
from model import UnetTMO
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


def save_image(im, p):
    base_dir = os.path.split(p)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(im, p)


@MODEL_REGISTRY
class PSENet(LightningModule):
    def __init__(
        self,
        tv_w,
        cc_w,
        recon_w,
        smooth_w,
        gamma_lower,
        gamma_upper,
        number_refs,
        lr,
        afifi_evaluation=False,
    ):
        super().__init__()
        self.tv_w = tv_w
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper
        self.number_refs = number_refs
        self.afifi_evaluation = afifi_evaluation
        self.lr = lr
        self.model = UnetTMO()
        self.mse = torch.nn.MSELoss()
        self.loss = Loss(tv_w, cc_w, recon_w, smooth_w)
        self.iqa = IQA()
        self.saved_input = None
        self.saved_pseudo_gt = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=[0.9, 0.99]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "total_loss",
        }

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["total_loss"])

    def cv2_to_tensor(self, cv2_img):
        cv2_img = cv2_img[:, :, ::-1]
        cv2_img = cv2_img / 255.0
        return torch.from_numpy(cv2_img).float().permute(2, 0, 1)

    def tensor_to_cv2(self, tensor_img):
        tensor_img = tensor_img.detach().cpu()
        transform = transforms.ToPILImage()
        pil_img = transform(tensor_img)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def cover_lights(self, syn_pseudo, syn_dark):
        cv2_pseudo = self.tensor_to_cv2(syn_pseudo)
        cv2_dark = self.tensor_to_cv2(syn_dark)

        cv2_dark_gray = cv2.cvtColor(cv2_dark, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            cv2_dark_gray, thresh=225, maxval=255, type=cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img_pseudo = cv2_pseudo.copy()

        # img_dark = cv2_dark.copy()
        # img_dark[(mask==255).all(-1)] = [0,0,0]
        # cv2_dark_w = cv2.addWeighted(img_dark, 0.4, cv2_dark, 0.6, 0, img_dark)

        img_pseudo[(mask_bgr == 255).all(-1)] = [128, 128, 128]
        cv2_pseudo_w = cv2.addWeighted(img_pseudo, 0.1, cv2_pseudo, 0.9, 0, img_pseudo)
        # return self.cv2_to_tensor(cv2_pseudo_w), self.cv2_to_tensor(cv2_dark_w)
        return self.cv2_to_tensor(cv2_pseudo_w)

    def generate_pseudo_gt(self, im):
        bs, ch, h, w = im.shape
        underexposed_ranges = torch.linspace(
            0, self.gamma_upper, steps=self.number_refs + 1
        ).to(im.device)[:-1]
        step_size = self.gamma_upper / self.number_refs
        underexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * step_size
            + underexposed_ranges[None, :]
        )
        overrexposed_ranges = torch.linspace(
            self.gamma_lower, 0, steps=self.number_refs + 1
        ).to(im.device)[:-1]
        step_size = -self.gamma_lower / self.number_refs
        overrexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device)
            * overrexposed_ranges[None, :]
        )
        gammas = torch.cat([underexposed_gamma, overrexposed_gamma], dim=1)
        # gammas: [bs, nref], im: [bs, ch, h, w] -> synthetic_references: [bs, nref, ch, h, w]
        synthetic_references = 1 - (1 - im[:, None]) ** gammas[:, :, None, None, None]
        syn_dark = synthetic_references[:, -1, :, :, :]
        previous_iter_output = self.model(im)[0].clone().detach()
        references = torch.cat(
            [im[:, None], previous_iter_output[:, None], synthetic_references], dim=1
        )
        nref = references.shape[1]
        scores = self.iqa(references.view(bs * nref, ch, h, w))
        scores = scores.view(bs, nref, 1, h, w)
        max_idx = torch.argmax(scores, dim=1)
        max_idx = max_idx.repeat(1, ch, 1, 1)[:, None]
        pseudo_gt = torch.gather(references, 1, max_idx)
        syn_pseudo = pseudo_gt.squeeze(1)
        for idx, dark_ref in enumerate(syn_dark):
            pseudo_ref = self.cover_lights(syn_pseudo[idx], dark_ref)
            syn_pseudo[idx] = pseudo_ref
        return pseudo_gt.squeeze(1), syn_pseudo.squeeze(1)  # pseudo_gt, syn_pseudo

    def training_step(self, batch, batch_idx):
        # a hack to get the output from previous iterator
        # In nth interator, we use (n - 1)th batch instead of n-th batch to update model's weight. n-th batch will be used to generate a pseudo gt with current model's weigh and then is saved to use in (n + 1)th interator

        # saving n-th input and n-th pseudo gt
        nth_input = batch
        nth_pseudo_gt, syn_pseudo = self.generate_pseudo_gt(batch)
        if self.saved_input is not None:
            # getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handeled automatically by pytorch lightning)
            im = self.saved_input
            pred_im, pred_gamma = self.model(im)
            pseudo_gt = self.saved_pseudo_gt
            loss = self.loss.get_loss(pred_im, pred_gamma, pseudo_gt, syn_pseudo)

            # logging
            self.log(
                "train_loss/",
                {"loss": loss},
                on_epoch=True,
                on_step=False,
            )
            self.log("total_loss", loss, on_epoch=True, on_step=False)
            if batch_idx == 0:
                visuals = [im, pseudo_gt, pred_im]
                visuals = torchvision.utils.make_grid([v[0] for v in visuals])
                self.logger.experiment.add_image("images", visuals, self.current_epoch)
        else:
            # skip updating model's weight at the first batch
            loss = None
            self.log("total_loss", 0, on_epoch=True, on_step=False)
        # saving n-th input and n-th pseudo gt
        self.saved_input = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            pred_im, pred_gamma = self.model(batch)
            self.logger.experiment.add_images("val_input", batch, self.current_epoch)
            self.logger.experiment.add_images("val_output", pred_im, self.current_epoch)

    def test_step(self, batch, batch_idx, test_idx=0):
        input_im, path = batch[0], batch[-1]
        pred_im, pred_gamma = self.model(input_im)
        for i in range(len(path)):
            save_image(pred_im[i], os.path.join(self.logger.log_dir, path[i]))

        if len(batch) == 3:
            gt = batch[1]
            psnr = piq.psnr(pred_im, gt)
            ssim = piq.ssim(pred_im, gt)
            self.log("psnr", psnr, on_step=False, on_epoch=True)
            self.log("ssim", ssim, on_step=False, on_epoch=True)
            if self.afifi_evaluation:
                assert len(path) == 1, "only support with batch size 1"
                if "N1." in path[0]:
                    self.log("psnr_under", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_under", ssim, on_step=False, on_epoch=True)
                else:
                    self.log("psnr_over", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_over", ssim, on_step=False, on_epoch=True)
