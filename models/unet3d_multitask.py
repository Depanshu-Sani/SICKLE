import torch
import torch.nn as nn


def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, timesteps, dropout):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes

        feats = 4
        self.en3 = conv_block(in_channel, feats * 4, feats * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * 4, feats * 8, feats * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.center_in = center_in(feats * 8, feats * 16)
        self.center_out = center_out(feats * 16, feats * 8)
        self.dc4 = conv_block(feats * 16, feats * 8, feats * 8)
        self.trans3 = up_conv_block(feats * 8, feats * 4)
        self.dc3 = conv_block(feats * 8, feats * 4, feats * 2)
        self.final_ct = nn.Conv3d(feats * 2, n_classes, kernel_size=3, stride=1, padding=1)
        self.final_sd = nn.Conv3d(feats * 2, 1, kernel_size=3, stride=1, padding=1)
        self.final_td = nn.Conv3d(feats * 2, 1, kernel_size=3, stride=1, padding=1)
        self.final_hd = nn.Conv3d(feats * 2, 1, kernel_size=3, stride=1, padding=1)
        self.final_cy = nn.Conv3d(feats * 2, 1, kernel_size=3, stride=1, padding=1)
        self.fn_ct = nn.Linear(timesteps, 1)
        self.fn_sd = nn.Linear(timesteps, 1)
        self.fn_td = nn.Linear(timesteps, 1)
        self.fn_hd = nn.Linear(timesteps, 1)
        self.fn_cy = nn.Linear(timesteps, 1)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = x.cuda()
        en3 = self.en3(x)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        center_in = self.center_in(pool_4)
        center_out = self.center_out(center_in)
        concat4 = torch.cat([center_out, en4], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3], dim=1)
        dc3 = self.dc3(concat3)

        final_ct = self.final_ct(dc3)
        final_ct = final_ct.permute(0, 1, 3, 4, 2)
        shape_num = final_ct.shape[0:4]
        final_ct = final_ct.reshape(-1, final_ct.shape[4])
        final_ct = self.dropout(final_ct)
        final_ct = self.fn_ct(final_ct)
        final_ct = final_ct.reshape(shape_num)

        final_sd = self.final_sd(dc3)
        final_sd = final_sd.permute(0, 1, 3, 4, 2)
        shape_num = final_sd.shape[0:4]
        final_sd = final_sd.reshape(-1, final_sd.shape[4])
        final_sd = self.dropout(final_sd)
        final_sd = self.fn_sd(final_sd)
        final_sd = final_sd.reshape(shape_num)

        final_td = self.final_td(dc3)
        final_td = final_td.permute(0, 1, 3, 4, 2)
        shape_num = final_td.shape[0:4]
        final_td = final_td.reshape(-1, final_td.shape[4])
        final_td = self.dropout(final_td)
        final_td = self.fn_td(final_td)
        final_td = final_td.reshape(shape_num)

        final_hd = self.final_hd(dc3)
        final_hd = final_hd.permute(0, 1, 3, 4, 2)
        shape_num = final_hd.shape[0:4]
        final_hd = final_hd.reshape(-1, final_hd.shape[4])
        final_hd = self.dropout(final_hd)
        final_hd = self.fn_hd(final_hd)
        final_hd = final_hd.reshape(shape_num)

        final_cy = self.final_cy(dc3)
        final_cy = final_cy.permute(0, 1, 3, 4, 2)
        shape_num = final_cy.shape[0:4]
        final_cy = final_cy.reshape(-1, final_cy.shape[4])
        final_cy = self.dropout(final_cy)
        final_cy = self.fn_cy(final_cy)
        final_cy = final_cy.reshape(shape_num)
        # final = self.logsoftmax(final)

        return {"crop_type": final_ct,
                "sowing_date": final_sd,
                "transplanting_date": final_td,
                "harvesting_date": final_hd,
                "crop_yield": final_cy,
                }