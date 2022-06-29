import torch
import torch.nn as nn
from collections import defaultdict
from protors.components import UnionLayer, LRLayer, THRESHOLD

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
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad, is_con=(i%2==1))
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

    def binarized_forward(self, x, explain_info:dict=None):
        with torch.no_grad():
            x_res = None
            rid_list = [] # ids and weights of matched rules
            for i, layer in enumerate(self.layer_list):
                if i < 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x # skip connection

                    # extra local explanation code
                    # record extra information and put it in explain_info
                    if explain_info is not None and i == len(self.layer_list) - 1: 
                        # iterate through the samples
                        for instance in x_cat:
                            rids = defaultdict(lambda: list()) # set of matched rules for this instance
                            for node_idx, activation in enumerate(instance):
                                if activation > THRESHOLD: # this node is activated 
                                    # get rule id and put in list
                                    offset = -1 if node_idx < self.layer_list[i-1].output_dim else -2 # preceding layer or skip connection layer
                                    rid = self.layer_list[i + offset].dim2id[node_idx]
                                    if rid != -1: 
                                        rids[(offset, rid)].append(node_idx) 
                            rid_list.append(rids)
                        
                        # save information
                        explain_info['matched_rules'] = rid_list

                    # the MAIN forwarding code
                    x_res = x
                    x = layer.binarized_forward(x_cat)

            return x

