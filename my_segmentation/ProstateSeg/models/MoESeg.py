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


class GatedFusionModule(nn.Module):
    def __init__(self, input_channels):
        super(GatedFusionModule, self).__init__()

        # Convolutional layers for input transformation
        self.conv_x = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv_y = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1)

        # Gating mechanism
        self.gate = nn.Conv3d(input_channels*2, 1, kernel_size=1)

    def forward(self, x, y):
        # Apply convolutional transformation to x and y
        transformed_x = self.conv_x(x)
        transformed_y = self.conv_y(y)

        # Concatenate the transformed inputs
        fused_input = torch.cat((transformed_x, transformed_y), dim=1)

        # Apply gating mechanism
        gate_output = torch.sigmoid(self.gate(fused_input))

        # Element-wise multiplication of gated output with transformed_y
        fused_output = gate_output * transformed_y

        # Return the fused output
        return fused_output

class CrossAttentionFusionModule(nn.Module):
    def __init__(self, input_channels):
        super(CrossAttentionFusionModule, self).__init__()

        # Convolutional layers for query, key, and value transformations
        self.query_transform = nn.Conv3d(input_channels, input_channels, kernel_size=1)
        self.key_transform = nn.Conv3d(input_channels, input_channels, kernel_size=1)
        self.value_transform = nn.Conv3d(input_channels, input_channels, kernel_size=1)

        # Softmax activation
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y):
        # Apply convolutional transformations to x and y
        query = self.query_transform(x)
        key = self.key_transform(y)
        value = self.value_transform(y)

        # Reshape query, key, and value for matrix multiplication
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), -1)

        # Compute attention scores
        attention_scores = torch.matmul(query, key)
        attention_weights = self.softmax(attention_scores)

        # Weighted sum of values using attention weights
        fused_output = torch.matmul(attention_weights, value)

        # Reshape fused_output back to 3D tensor
        fused_output = fused_output.view(fused_output.size(0), fused_output.size(1), *y.size()[2:])

        # Return the fused output
        return fused_output

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


class UnetDecoder(nn.Module):
    def __init__(self, n_class=2, act='relu') -> None:
        super(UnetDecoder, self).__init__()
        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        # output layer
        self.final_conv = nn.Conv3d(64, n_class, kernel_size=1)

    def forward(self, out512, skip_out256, skip_out128, skip_out64):
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64  = self.up_tr64(out_up_128, skip_out64)
        out = self.final_conv(out_up_64)
        return out

        
class MoESeg(nn.Module):
    def __init__(self, n_class=2, act='relu'):
        super(MoESeg, self).__init__()

        # encoder net
        self.encoder_t2w = UnetEncoder()
        self.encoder_adcdwi = UnetEncoder()

        # feature fusion layer
        self.gatedfusion = GatedFusionModule(input_channels=512)
        self.crossattfusion = CrossAttentionFusionModule(input_channels=512)

        # decoder net
        self.decoder_t2w = UnetDecoder(n_class=n_class)
        self.decoder_adc = UnetDecoder(n_class=n_class)
        self.decoder_dwi = UnetDecoder(n_class=n_class)

        # moe gate
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0),
        )
        self.w_gating = nn.Parameter(torch.randn(256, 3), requires_grad=True)

    def forward(self, x, inference=False, modality=0):
        """
        x.shape: B, 3, 128, 128, 16
        """
        t2w = x[:,0,...].unsqueeze(1)
        adc = x[:,1,...].unsqueeze(1)
        dwi = x[:,2,...].unsqueeze(1)
        
        # encoder all these features
        t2w_skip_out64, t2w_skip_out128, t2w_skip_out256, t2w_out512 = self.encoder_t2w(t2w)
        adc_skip_out64, adc_skip_out128, adc_skip_out256, adc_out512 = self.encoder_adcdwi(adc)
        dwi_skip_out64, dwi_skip_out128, dwi_skip_out256, dwi_out512 = self.encoder_adcdwi(dwi)
        
        # fuse all features
        skip_out64  = (t2w_skip_out64  + adc_skip_out64  + dwi_skip_out64)
        skip_out128 = (t2w_skip_out128 + adc_skip_out128 + dwi_skip_out128)
        skip_out256 = (t2w_skip_out256 + adc_skip_out256 + dwi_skip_out256)
        dwiadc_fusion = self.gatedfusion(dwi_out512, adc_out512)
        out512      = self.crossattfusion(dwiadc_fusion, t2w_out512)
        
        # decoder the feature
        t2w_out = self.decoder_t2w(out512, skip_out256, skip_out128, skip_out64)
        dwi_out = self.decoder_dwi(out512, skip_out256, skip_out128, skip_out64)
        adc_out = self.decoder_adc(out512, skip_out256, skip_out128, skip_out64)

        # decision-level fusion (moe)
        flatten_feature = self.GAP(out512)
        flatten_feature = flatten_feature.view(flatten_feature.shape[0], 256)
        raw_gates = torch.einsum("bc,cg->bg", flatten_feature, self.w_gating)
        raw_gates = nn.functional.sigmoid(raw_gates)
        raw_gates = torch.where(raw_gates>0.5, 1, 0).float()  # B, 3
        
        expert_outputs = torch.stack([t2w_out, adc_out, dwi_out], dim=1)
        moe_out = torch.einsum("bnchwd,bn->bchwd", expert_outputs, raw_gates)
        moe_out = torch.where(moe_out>=1, 1, 0)   # B, num_cls, w, h, d
        moe_out = nn.functional.one_hot(moe_out[:,1,...], num_classes=2).permute(0, 4, 1, 2, 3).float() # B, num_cls, w, h, d
        
        if inference:
            out_list = [t2w_out, dwi_out, adc_out, moe_out]
            print(raw_gates, end='\t')
            return out_list[modality]
        
        return t2w_out, dwi_out, adc_out, moe_out
        



if __name__ == "__main__":
    x = torch.ones((2, 3, 128, 128, 16))
    model = MoESeg()
    
    # load pretrain model
    pred = model(x)
    print(pred.shape)