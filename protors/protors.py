import argparse
import os
import pickle
from sklearn.preprocessing import binarize
import sys
from collections import defaultdict

import torch
import torch.nn as nn

from protors.mllp import MLLP
from protors.prototype_sim import Similarity, Binarization

class ProtoRS(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module = nn.Identity()
                 ):
        super().__init__()
        self.num_classes = num_classes
        # Conv net
        self.net = feature_net
        self.add_on = add_on_layers
        # Prototype layer
        self.epsilon = 1e-4
        self.num_prototypes = args.num_prototypes
        self.prototype_shape = (args.W1, args.H1, args.num_features)
        self.prototype_layer = FocalSimilarity(self.num_prototypes,
                                        args.num_features,
                                        args.W1,
                                        args.H1,
                                        self.epsilon)
        self.binarize_layer = Binarization(self.num_prototypes)
        # MLLP
        self.rs_dim_list = [self.num_prototypes] + \
                            list(map(int, args.structure.split('@'))) + \
                            [self.num_classes]
        self.mllp = MLLP(dim_list=self.rs_dim_list, 
                        estimated_grad=args.estimated_grad)

        # Placeholder for the final rule set
        self.final_dim2id = {}
    
    @property
    def features_requires_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_requires_grad.setter
    def features_requires_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_requires_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_requires_grad.setter
    def add_on_layers_requires_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    @property
    def prototypes_requires_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_requires_grad.setter
    def prototypes_requires_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def binarization_requires_grad(self) -> bool:
        return self.binarize_layer.thresholds.requires_grad

    @binarization_requires_grad.setter
    def binarization_requires_grad(self, val: bool):
        self.binarize_layer.thresholds.requires_grad = val

    @property
    def mllp_requires_grad(self) -> bool:
        return self.mllp.layer_list[-1].requires_grad

    @mllp_requires_grad.setter
    def mllp_requires_grad(self, val: bool):
        for layer in self.mllp.layer_list:
            layer.requires_grad = val

    def save(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)
        # Save the out_map of the model to the target directory
        with open(directory_path + '/model_pickle.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')  

    def forward(self, 
                xs: torch.Tensor,
                explain_info: dict = None) -> tuple:
        # Forward conv net
        features = self.net(xs)
        features = self.add_on(features)
        bs, D, W, H = features.shape
        # Compute similarities
        similarities = self.prototype_layer(features, W, H).view(bs, self.num_prototypes)
        similarities_cont = self.binarize_layer(similarities)
        similarities_disc = self.binarize_layer.binarized_forward(similarities, explain_info=explain_info)
        # Classify
        out_cont = self.mllp(similarities_cont)
        out_disc = self.mllp.binarized_forward(similarities_disc, explain_info=explain_info)
        return out_cont, out_disc

    def forward_partial(self,
                        xs: torch.Tensor
                        ) -> tuple:
        # Forward conv net
        features = self.net(xs)
        features = self.add_on(features)
        # Compute similarities
        similarities = self.prototype_layer(features, 1, 1)
        return features, similarities

    # adapted from RRL
    def detect_dead_node(self, data_loader=None, device_id=torch.device('cpu')): # change self.net to self.mllp and add argument 'device'
        with torch.no_grad():
            for layer in self.mllp.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=device_id)
                layer.forward_tot = 0
            for x, y in data_loader:
                # forward up until mllp
                x = x.cuda(device_id)
                features = self.net(x)
                features = self.add_on(features)
                bs, D, W, H = features.shape
                similarities = self.prototype_layer(features, W, H).view(bs, self.num_prototypes)
                x = self.binarize_layer.binarized_forward(similarities)

                # record activation numbers
                x_res = None # x residual / skip connection
                for i, layer in enumerate(self.mllp.layer_list[:-1]):
                    if i <= 1:
                        x = layer.binarized_forward(x)
                    else:
                        x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                        x_res = x
                        x = layer.binarized_forward(x_cat)
                    layer.node_activation_cnt += torch.sum(x, dim=0)
                    layer.forward_tot += x.shape[0]

    def rule_print(self, label_name, train_loader, file=sys.stdout, device=torch.device('cpu')):
        # FIXME: these hasattr() are just hotfixes, add these attributes to the corresponding layer
        if not hasattr(self.binarize_layer, 'dim2id'):
            self.binarize_layer.__setattr__('dim2id', {i: i for i in range(self.num_prototypes)})
        if not hasattr(self.binarize_layer, 'layer_type'):
            self.binarize_layer.__setattr__('layer_type', 'binarization')
        layer_list = nn.ModuleList([self.binarize_layer]) + self.mllp.layer_list # imo this is better then reindexing all the layers

        # prune duplicate prototype
        # TODO: add this snippet into a seperate function
        print('[+] Pruning duplicate prototypes...')
        for i in range(self.num_prototypes):
            if self.binarize_layer.dim2id[i] != i: 
                continue
            for j in range(i+1, self.num_prototypes):
                if self.prototype_layer.prototype_vectors[i].equal(self.prototype_layer.prototype_vectors[j]):
                    self.binarize_layer.dim2id[j] = i

        # pruning/detect dead nodes
        if layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")
        print('[+] Detecting dead nodes...')
        if layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader, device)

        # extract rules from binarization layer -> first logical layer
        print('[+] Extracting rules from model...')
        labels = self.prototype_layer.get_prototype_labels()
        layer_list[1].get_rules(layer_list[0], None)
        layer_list[1].get_rule_description((None, labels))

        # the second logical layer (if exists) has no skip connections
        if len(layer_list) >= 4:
            layer_list[2].get_rules(layer_list[1], None)
            layer_list[2].get_rule_description((None, layer_list[1].rule_name), wrap=True)

        # network with 3 hidden layers and more has skip connections 
        if len(layer_list) >= 5:
            for i in range(3, len(layer_list) - 1):
                layer_list[i].get_rules(layer_list[i - 1], layer_list[i - 2])
                layer_list[i].get_rule_description(
                    (layer_list[i - 2].rule_name, layer_list[i - 1].rule_name), wrap=True)

        # get the dim2id dictionary of the linear layer by merging the dim2id of the last logical layer + the skip connection layer 
        prev_layer = layer_list[-2]
        skip_connect_layer = layer_list[-3]
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot) # boolean for 'No dead node'
        if skip_connect_layer.layer_type == 'union':
            # similar code to UnionLayer's get_rules()
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        else:
            merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

        Wl, bl = list(layer_list[-1].parameters()) # weights and biases of the last layer a.k.a the linear layer
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {} # key: rule id, value: node id of the last logical layer or the last skip connection layer
        # iterate over the nodes/labels
        for label_id, wl in enumerate(Wl):
            # iterate over the weights of each label
            for i, w in enumerate(wl):
                rid = merged_dim2id[i] # look up rule ID from the merged dictionary
                if rid == -1 or rid[1] == -1: # invalid rule id/ rule with dead nodes
                    continue
                marked[rid][label_id] += w # combine duplicate rules for this label by adding up their weights
                rid2dim[rid] = i % prev_layer.output_dim # this rule corresponds to the node i from the last logical layer/skip connection layer

        # print rules
        kv_list = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True) # sort rules by abs(weights) a.k.a rule significance
        print('[+] Printing {} rule(s)...'.format(len(kv_list)))
        print('RID', end='\t', file=file)
        for i, ln in enumerate(label_name):
            print('{}(b={:.4f})'.format(ln, bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        for k, v in kv_list:
            rid = k
            print(rid, end='\t', file=file)
            for li in range(len(label_name)):
                print('{:.4f}'.format(v[li]), end='\t', file=file)
            now_layer = layer_list[-1 + rid[0]]
            # print('({},{})'.format(now_layer.node_activation_cnt[rid2dim[rid]].item(), now_layer.forward_tot))
            print('{:.4f}'.format((now_layer.node_activation_cnt[rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return kv_list