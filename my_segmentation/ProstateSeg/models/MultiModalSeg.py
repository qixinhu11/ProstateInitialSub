import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            # out_before_pool = out
            return out 
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
            return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class GatedFusion(nn.Module):
    """
    Gated Fusion for two features
    """
    def __init__(self, ) -> None:
        super(GatedFusion, self).__init__()


class UnetEncoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, act='relu'):
        super(UnetEncoder, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        out64,  skip_out64 = self.down_tr64(x)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out512 = self.down_tr512(out256)

        return (skip_out64, skip_out128, skip_out256, out512)
        

class MultiModalSeg(nn.Module):
    def __init__(self, n_class=2, act='relu'):
        super(MultiModalSeg, self).__init__()

        # encoder net
        self.encoder_t2w = UnetEncoder()
        self.encoder_adcdwi = UnetEncoder()

        # feature fusion layer
        
        # decoder net
        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)

        # output layer
        self.final_conv = nn.Conv3d(64, n_class, kernel_size=1)

    def forward(self, x):
        """
        x.shape: B, 3, 96, 96, 48
        """
        t2w = x[:,0,...].unsqueeze(1)
        adc = x[:,1,...].unsqueeze(1)
        dwi = x[:,2,...].unsqueeze(1)

        # encoder all these features
        t2w_skip_out64, t2w_skip_out128, t2w_skip_out256, t2w_out512 = self.encoder_t2w(t2w)
        adc_skip_out64, adc_skip_out128, adc_skip_out256, adc_out512 = self.encoder_adcdwi(adc)
        dwi_skip_out64, dwi_skip_out128, dwi_skip_out256, dwi_out512 = self.encoder_adcdwi(dwi)
        
        # simple fusion(add) all features
        skip_out64  = (t2w_skip_out64  + adc_skip_out64  + dwi_skip_out64)
        skip_out128 = (t2w_skip_out128 + adc_skip_out128 + dwi_skip_out128)
        skip_out256 = (t2w_skip_out256 + adc_skip_out256 + dwi_skip_out256)
        out512      = (t2w_out512 + adc_out512 + dwi_out512)

        # decoder the feature
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64  = self.up_tr64(out_up_128, skip_out64)
        out = self.final_conv(out_up_64)

        return out
        



if __name__ == "__main__":
    x = torch.ones((1, 3, 96, 96, 48))
    model = MultiModalSeg()
    
    # load pretrain model
    pred = model(x)
    print(pred.shape)