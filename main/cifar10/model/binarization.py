import math
import torch 
import torch.nn as nn

"""
Function for activation binarization
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        zero_index = torch.abs(input) > 1
        grad_input = grad_output.clone()
        return grad_input*zero_index


class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        # self.bias = nn.Parameter(torch.Tensor(out_size))
        self.bias = None
        self.threshold = nn.Parameter(torch.Tensor(out_size)) #This was it
        self.step = BinaryStep.apply #it becomes forward embedded with specified custom backward
        self.mask = torch.ones(out_size, in_size)
        self.ratio = 1
        self.reset_parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zero_count = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0)

    def mask_generation(self, weight, thresholds):
        abs_weight = torch.abs(self.weight)
        abs_weight_mean = torch.mean(abs_weight, 1)
        abs_weight_mean = abs_weight_mean.view(abs_weight.shape[0], -1)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight_mean - threshold
        abs_weight = abs_weight.repeat(1, weight.shape[1])
        mask = self.step(abs_weight)
        self.mask = mask.to(self.device)
        self.ratio = torch.sum(self.mask) / self.mask.numel()

    def forward(self, input):
        self.threshold.clip(-5, 5)
        if self.ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0.)
                self.zero_count +=1
            self.mask_generation(self.weight, self.threshold)

        self.mask_generation(self.weight, self.threshold) #generate binary masks    
        masked_weight = self.weight * self.mask
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

    


class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c 
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups
        self.mask = torch.ones((out_c, in_c //groups, *kernel_size))
        self.ratio =1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = 1
        self.zero_count = 0

        ## define weight 
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_c))
        self.step = BinaryStep.apply
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def mask_generation(self, weight, thresholds):
        weight_shape = self.weight.shape 
        threshold = self.threshold.view(weight_shape[0], -1) #make it like a column
        abs_weight = torch.abs(self.weight)
        abs_weight = abs_weight.view(weight_shape[0], -1) 
        weight_mean = torch.mean(abs_weight, 1) #average of weights of each filter
        weight_mean = weight_mean.view(weight_shape[0], -1) #makie it like a column
        weight_mean = weight_mean.repeat(1, weight_shape[1]*weight_shape[2]**2) #repeat for every weight
        weight_mean = weight_mean - threshold #automatically does filter-wise calculation

        mask = self.step(weight_mean)
        mask = mask.view(weight_shape)
        self.mask = mask.to(self.device)
        self.ratio = torch.sum(self.mask) / self.mask.numel()

    def forward(self, x):
        self.threshold.clip(-5, 5)
        if self.ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
 
            self.mask_generation(self.weight, self.threshold)
            self.zero_count += 1

        self.mask_generation(self.weight, self.threshold) #generate binary masks
        masked_weight = self.weight * self.mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        self.output_dim = conv_out.shape[-1] #used for FLOP calculations

        return conv_out
