#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import reduce
from os.path import realpath

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.nn import QuantHardTanh, QuantReLU

from pyverilator import PyVerilator

from logicnets.quant import QuantBrevitasActivation
from logicnets.nn import SparseLinearNeq, ScalarBiasScale, RandomFixedSparsityMask2D, RandomFixedSparsityMask3D, SparseConv1dNeq

from logicnets.init import random_restrict_fanin

import torch.nn.functional as F #ALI

class AtrialFibrillationNeqModel(nn.Module):
    def __init__(self, model_config):
        super(AtrialFibrillationNeqModel, self).__init__()
        self.model_config = model_config
        self.num_neurons = [model_config["input_length"]] + model_config["hidden_layers"] + [model_config["output_length"]]

        # print([model_config["input_length"]]) # [5250] #all of these prints are for the jsc-s config
        # print(model_config["hidden_layers"]) # [64, 32, 32, 32]
        # print([model_config["output_length"]]) # [2]
        # print(self.num_neurons) # [5250, 64, 32, 32, 32, 2]
        # print(len(self.num_neurons))
        # input()

        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i-1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            if i == 1: #Ali: Input layer
                bn_in = nn.BatchNorm1d(in_features)
                input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
                input_quant = QuantBrevitasActivation(QuantHardTanh(model_config["input_bitwidth"], max_val=1., narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn_in, input_bias])
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["input_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=input_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask})
                layer_list.append(layer)
            elif i == len(self.num_neurons)-1: #Ali: output layer
                output_bias_scale = ScalarBiasScale(bias_init=0.33)
                output_quant = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, narrow_range=False, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn], post_transforms=[output_bias_scale])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
            else: #Ali: all hidden layers
                output_quant = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
                mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
                layer = SparseLinearNeq(in_features, out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quant, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
                layer_list.append(layer)
        self.module_list = nn.ModuleList(layer_list)
        # print(len(self.module_list)) # 5
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    def verilog_inference(self, verilog_dir, top_module_filename, logfile: bool = False, add_registers: bool = False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}", verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.logfile = logfile
        if add_registers:
            self.latency = len(self.num_neurons)

    def pytorch_inference(self):
        self.is_verilog_inference = False

    def verilog_forward(self, x):
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features*input_bitwidth
        total_output_bits = self.module_list[-1].out_features*output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i,:]
            y_i = self.pytorch_forward(x[i:i+1,:])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a,b: a+b, xv_i[::-1])
            ysc_i = reduce(lambda a,b: a+b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                #print(self.dut.io.M5)
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i,2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert(expected == result)
            res_split = [result[i:i+output_bitwidth] for i in range(0, len(result), output_bitwidth)][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i,:] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(f"{int(xvc_i,2):0{int(total_input_bits)}b}{int(ysc_i,2):0{int(total_output_bits)}b}\n")
        return y

    def pytorch_forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)

class AtrialFibrillationLutModel(AtrialFibrillationNeqModel):
    pass

class AtrialFibrillationVerilogModel(AtrialFibrillationNeqModel):
    pass


class AtrialFibrillationConv1dNeqModel(nn.Module):
    def __init__(self, model_config):
        super(AtrialFibrillationConv1dNeqModel, self).__init__()
        # self.model_config = model_config
        # self.num_neurons = [model_config["input_length"]] + model_config["hidden_layers"] + [model_config["output_length"]]
        layer_list = []

        #Model Architecture: [conv1d_1: in:1  out:8,
        #                     conv1d_2: in:8    out:16,
        #                     conv1d_3: in:16    out:32,
        #                     linear1:  in:32*41   out: 32
        #                     linear2:  in:32   out:2   ]

        #conv1d_1 layer
        in_channels = 1 #model_config["input_length"]
        out_channels = 8
        bn = nn.BatchNorm1d(out_channels)
        bn_in = nn.BatchNorm1d(in_channels)
        input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
        input_quant = QuantBrevitasActivation(QuantHardTanh(model_config["input_bitwidth"], max_val=1., narrow_range=False, quant_type=QuantType.INT,
                          scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn_in, input_bias]) # pre_transforms=[bn_in, input_bias]

        output_quant_1 = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT,
                      scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])

        mask = RandomFixedSparsityMask3D(in_channels, out_channels, fan_in=model_config["input_fanin"]) #kernel_size = 5 default for now

        layer = SparseConv1dNeq(in_channels, out_channels, input_quant=input_quant, output_quant=output_quant_1,sparse_linear_kws={'mask': mask})
        layer_list.append(layer)

        #conv1d_2 layer
        in_channels = 8
        out_channels = 16
        bn = nn.BatchNorm1d(out_channels)
        output_quant_2 = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT,
                      scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])

        mask = RandomFixedSparsityMask3D(in_channels, out_channels, fan_in=model_config["hidden_fanin"])

        layer = SparseConv1dNeq(in_channels, out_channels, input_quant=output_quant_1, output_quant=output_quant_2,sparse_linear_kws={'mask': mask})
        layer_list.append(layer)

        #conv1d_3 layer
        in_channels = 16
        out_channels = 32
        bn = nn.BatchNorm1d(out_channels)
        output_quant_3 = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT,
                      scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
        mask = RandomFixedSparsityMask3D(in_channels, out_channels, fan_in=model_config["hidden_fanin"])
        layer = SparseConv1dNeq(in_channels, out_channels, input_quant=output_quant_2, output_quant=output_quant_3,
                                sparse_linear_kws={'mask': mask})
        layer_list.append(layer)
        #shape after passing through the conv layers: output shape =  torch.Size([batchsize, 32, 41]) < after maxpool1d with 5 filter size

        # linear1 layer
        in_features = 32*41 #Transitioning from a conv1d layer to a linear layer. it has to be 5238(without maxpool1d)/32*41(with maxpool1d filter size 5), else: RuntimeError: size mismatch, m1: [32768 x 5238], m2: [32 x 64]
        out_features = 32
        bn = nn.BatchNorm1d(out_features)
        output_quant_4 = QuantBrevitasActivation(QuantReLU(bit_width=model_config["hidden_bitwidth"], max_val=1.61, quant_type=QuantType.INT,
                      scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn])
        mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["hidden_fanin"])
        layer = SparseLinearNeq(in_features, out_features, input_quant=output_quant_3,
                                output_quant=output_quant_4, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
        layer_list.append(layer)
        #shape after passing through this layer: output shape =  torch.Size([1024, 32, 32])

        # linear2 layer (output layer)
        in_features = 32
        out_features = 2
        bn = nn.BatchNorm1d(out_features)
        output_bias_scale = ScalarBiasScale(bias_init=0.33)
        output_quant_5 = QuantBrevitasActivation(QuantHardTanh(bit_width=model_config["output_bitwidth"], max_val=1.33, narrow_range=False,quant_type=QuantType.INT, scaling_impl_type=ScalingImplType.PARAMETER), pre_transforms=[bn],post_transforms=[output_bias_scale])
        mask = RandomFixedSparsityMask2D(in_features, out_features, fan_in=model_config["output_fanin"])
        layer = SparseLinearNeq(in_features, out_features, input_quant=output_quant_4,
                                output_quant=output_quant_5, sparse_linear_kws={'mask': mask}, apply_input_quant=False)
        layer_list.append(layer)

        self.module_list = nn.ModuleList(layer_list)
        # print(len(self.module_list)) # 5
        self.is_verilog_inference = False
        self.latency = 1
        self.verilog_dir = None
        self.top_module_filename = None
        self.dut = None
        self.logfile = None

    #ALI:this function is called after you have a top level verilog file.
    # is the code for generating verilog SparseLinear layers gona be the same for conv1d layers ?
    # this function is called from neq2lut.py
    # it is simulating the verilog files and passing input in it
    def verilog_inference(self, verilog_dir, top_module_filename, logfile: bool = False,
                          add_registers: bool = False):
        self.verilog_dir = realpath(verilog_dir)
        self.top_module_filename = top_module_filename
        self.dut = PyVerilator.build(f"{self.verilog_dir}/{self.top_module_filename}",
                                     verilog_path=[self.verilog_dir], build_dir=f"{self.verilog_dir}/verilator")
        self.is_verilog_inference = True
        self.logfile = logfile
        if add_registers:
            #ALI: self.latency = len(self.num_neurons) num_neurons is not implemented in this model
            self.latency = len(self.module_list) + 1

    def pytorch_inference(self):
        self.is_verilog_inference = False

    # ALI: FORWARDING FROM DATASET INSIDE A PYVERILATOR SYNTHESIZED MODEL
    def verilog_forward(self, x):
        # Get integer output from the first layer
        input_quant = self.module_list[0].input_quant
        output_quant = self.module_list[-1].output_quant
        _, input_bitwidth = self.module_list[0].input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.module_list[-1].output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.module_list[0].in_features * input_bitwidth
        total_output_bits = self.module_list[-1].out_features * output_bitwidth
        num_layers = len(self.module_list)
        input_quant.bin_output()
        self.module_list[0].apply_input_quant = False
        y = torch.zeros(x.shape[0], self.module_list[-1].out_features)
        x = input_quant(x)
        self.dut.io.rst = 0
        self.dut.io.clk = 0
        for i in range(x.shape[0]):
            x_i = x[i, :]
            y_i = self.pytorch_forward(x[i:i + 1, :])[0]
            xv_i = list(map(lambda z: input_quant.get_bin_str(z), x_i))
            ys_i = list(map(lambda z: output_quant.get_bin_str(z), y_i))
            xvc_i = reduce(lambda a, b: a + b, xv_i[::-1])
            ysc_i = reduce(lambda a, b: a + b, ys_i[::-1])
            self.dut["M0"] = int(xvc_i, 2)
            for j in range(self.latency + 1):
                # print(self.dut.io.M5)
                res = self.dut[f"M{num_layers}"]
                result = f"{res:0{int(total_output_bits)}b}"
                self.dut.io.clk = 1
                self.dut.io.clk = 0
            expected = f"{int(ysc_i, 2):0{int(total_output_bits)}b}"
            result = f"{res:0{int(total_output_bits)}b}"
            assert (expected == result)
            res_split = [result[i:i + output_bitwidth] for i in range(0, len(result), output_bitwidth)][::-1]
            yv_i = torch.Tensor(list(map(lambda z: int(z, 2), res_split)))
            y[i, :] = yv_i
            # Dump the I/O pairs
            if self.logfile is not None:
                with open(self.logfile, "a") as f:
                    f.write(
                        f"{int(xvc_i, 2):0{int(total_input_bits)}b}{int(ysc_i, 2):0{int(total_output_bits)}b}\n")
        return y


    def pytorch_forward(self, x):
        #ALI's code:
        for i in range(len(self.module_list)):
            # print("forwarding through layer",i)
            if i < 3 : # layers 0,1,2 are conv1d layers
                x = self.module_list[i](x)
                x = F.max_pool1d(x,5)
            elif i == 3 : # transitioning from conv1d to linear
                x = x.view(-1, 32*41) # flatten the data # TODO: instead of hard coded value, use a variable that calculates the shape of the data before feeding to the linear layer
                # print(x.shape) # torch.Size([1024, 1312])
                x = self.module_list[i](x)
            else :
                x = self.module_list[i](x)
        return x
        # Original Code:
        # for l in self.module_list:
        #     x = l(x)
        # return x

    def forward(self, x):
        if self.is_verilog_inference:
            return self.verilog_forward(x)
        else:
            return self.pytorch_forward(x)

class AtrialFibrillationConv1dLutModel(AtrialFibrillationConv1dNeqModel):
    pass

class AtrialFibrillationConv1dVerilogModel(AtrialFibrillationConv1dNeqModel):
    pass


model_configure = {
    "input_length" : 5250,
    "output_length" : 2,
    "hidden_layers": [64,32],
    "input_bitwidth": 2,
    "hidden_bitwidth": 2,
    "output_bitwidth": 2,
    "input_fanin": 3,
    "hidden_fanin": 3,
    "output_fanin": 3,
    "weight_decay": 1e-3,
    "batch_size": 1024,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "seed": 2,
    "checkpoint": None
}

model = AtrialFibrillationNeqModel(model_configure)
print(model)
