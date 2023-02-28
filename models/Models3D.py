import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F


class ThresholdNet(nn.Module):
    def __init__(self, channels=8):
        super(ThresholdNet, self).__init__()
        self.logvar_layer = nn.Linear(channels, channels)
        self.mu_layer = nn.Linear(channels, channels)

    def forward(self, x):
        return self.mu_layer(x), self.logvar_layer(x)


class UnetBPL3D(nn.Module):
    def __init__(self,
                 in_ch=1,
                 width=8,
                 depth=4,
                 out_ch=2,
                 norm='in'
                 ):

        super(UnetBPL3D, self).__init__()
        self.segmentor = Unet3D(in_ch, width, depth, out_ch, norm=norm, side_output=True)
        self.encoder = ThresholdNet(width*2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, x):
        outputs_dict = self.segmentor(x)
        output, threshold_input = outputs_dict['segmentation'], outputs_dict['side_output']
        threshold_input = self.avg_pool(threshold_input)
        mu, logvar = self.encoder(torch.flatten(threshold_input, start_dim=1))

        # learnt_threshold = self.reparameterize(mu, logvar)
        # learnt_threshold = F.softplus(learnt_threshold)
        # learnt_threshold = torch.sigmoid(learnt_threshold)

        return {
            'segmentation': output,
            'mu': mu,
            'logvar': logvar
            }


# in_ch, width, classes, side_output=False
class Unet3D(nn.Module):
    def __init__(self,
                 in_ch=1,
                 width=8,
                 depth=3,
                 classes=2,
                 norm='in',
                 side_output=False):
        '''
        Args:
            in_ch:
            width:
            depth:
            classes:
            norm:
            side_output:
        '''
        super(Unet3D, self).__init__()

        assert depth > 1
        self.depth = depth

        self.side_output_mode = side_output
        self.first_encoder = single_conv(in_channels=in_ch, out_channels=width, step=1)

        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        for i in range(0, self.depth, 1):
            self.encoders.append(double_conv_3d(in_channels=width*(2**i), out_channels=width*(2**(i+1)), step=2, norm=norm))

        for i in range(self.depth, 0, -1):
            self.decoders.append(double_conv_3d(in_channels=width*(2**i)+width*(2**(i-1)), out_channels=width*(2**(i-1)), step=1, norm=norm))

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

        self.last_decoder = double_conv_3d(in_channels=width*2, out_channels=width*2, step=1, norm=norm)
        self.classification = nn.Conv3d(2*width, classes, 1, bias=True)

    def forward(self, x):

        encoder_features = []
        y = self.first_encoder(x)
        encoder_features.append(y)

        for i in range(len(self.encoders)):
            y = self.encoders[i](y)
            encoder_features.append(y)

        del encoder_features[-1]

        for i in range(len(encoder_features)):
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[i](y)

        y_e = encoder_features[0]
        y = torch.cat([y_e, y], dim=1)
        y = self.last_decoder(y)
        output = self.classification(y)

        if self.side_output_mode is False:
            return {'segmentation': output}
        else:
            return {'segmentation': output,
                    'side_output': y}


def single_conv(in_channels, out_channels, step):
    # single convolutional layers
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, (3, 3, 3), stride=step, padding=(1, 1, 1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def double_conv_3d(in_channels, out_channels, step, norm='in'):
    '''
    Args:
        in_channels:
        out_channels:
        step:
        norm:
    Returns:
    '''
    if norm == 'in':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )






