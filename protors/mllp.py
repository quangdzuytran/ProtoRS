import torch
import torch.nn as nn

from protors.components import UnionLayer, LRLayer

class MLLP(nn.Module):
    def __init__(self, dim_list, estimated_grad=False):
        super(MLLP, self).__init__()

        self.dim_list = dim_list
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 3:
                num += self.layer_list[-2].output_dim
            if i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            else:
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        return self.continuous_forward(x)

    def continuous_forward(self, x):
        x_res = None
        for i, layer in enumerate(self.layer_list):
            if i < 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i < 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x
