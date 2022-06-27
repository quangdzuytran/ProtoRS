import torch
import torch.nn as nn
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, use_not=False, left=None, right=None):
        super(BinarizeLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}

        self.register_buffer('left', left)
        self.register_buffer('right', right)

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)
                cr = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)
            else:
                cl = 3. * (2. * torch.rand(self.n, self.input_dim[1]) - 1.)
                cr = 3. * (2. * torch.rand(self.n, self.input_dim[1]) - 1.)
            self.register_buffer('cl', cl)
            self.register_buffer('cr', cr)

    def forward(self, x):
        if self.input_dim[1] > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            x = x.unsqueeze(-1)
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
            return torch.cat((x_disc, Binarize.apply(x - self.cl.t()).view(x.shape[0], -1),
                              1 - Binarize.apply(x - self.cr.t()).view(x.shape[0], -1)), dim=1)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            return self.forward(x)

    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(self.cl.data > self.right, self.right, self.cl.data)
            self.cl.data = torch.where(self.cl.data < self.left, self.left, self.cl.data)

            self.cr.data = torch.where(self.cr.data > self.right, self.right, self.cr.data)
            self.cr.data = torch.where(self.cr.data < self.left, self.left, self.cr.data)

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        # get names of discrete features
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        # if NOT operators are allowed, attach '~' to the names above, as separate features
        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append('~' + feature_name[i])
        # get names of (discretized) continuous features
        if self.input_dim[1] > 0: 
            # lowerbounds then upperbounds
            for c, op in [(self.cl, '>'), (self.cr, '<')]:
                c = c.detach().cpu().numpy() # list of all (lower or upper) bounds of all features
                # for each feature i
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i] # get feature i's name
                    # for each bound assigned to feature i
                    for j in ci:
                        if mean is not None and std is not None:
                            j = j * std[fi_name] + mean[fi_name] # inverse standardization
                        bound_name.append('{} {} {:.3f}'.format(fi_name, op, j)) # feature names + '<' or '>' + bound value
        return bound_name


class Product(torch.autograd.Function):
    """Tensor product function."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(torch.relu(X) + EPSILON), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(torch.relu(X) + EPSILON), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'

        self.fc1 = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x):
        return self.fc1(x)

    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(0.0, None)


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        A = 1 - self.Product.apply(1 - self.W)
        return self.Product.apply(1 - (1 - x).unsqueeze(-1) * self.W.t()) * A

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        Ab = 1 - torch.prod(1 - Wb, dim=1)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb.t(), dim=1) * Ab

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - self.Product.apply(1 - x.unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)

# TODO: move this function into ProtoRS itself, for encapsulation sake
def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    """Given 2 consecutive layers, generate rules conferred by the weights between them.
    A rule is a combination of nodes from previous layers, connected together by a node from the current layer
    This function was copied and modified from RRL, since there are no discrete features to worry about.
    Comments were also added to aid comprehension.

    Args:
        prev_layer (torch.nn.Module): the preceding layer
        skip_connect_layer (torch.nn.Module): the layer that was linked to the current layer by a skip connection
        layer (torch.nn.Module): the current layer
        pos_shift (int, optional): because conjunctive and disjunctive layer are concatenated together, this number is used to marked the start of the disjunctive layer. Defaults to 0.

    Returns:
        dim2id (defaultdict): a dictionary with key being the node index of the current layer, and value is an index of rule_list
        rule_list (list): a list of rules a.k.a different combinations of node indexes of the previous layers
    """
    dim2id = defaultdict(lambda: -1) # key: node id (after concatenation); value: rule index
    rules = {} # a dictionary to detect duplicate rules
    tmp = 0 # unique rules count, used to index the rules inside the rule_list
    rule_list = [] # the list of node_idx combinations to be return
    Wb = (layer.W > 0.5).type(torch.int).detach().cpu().numpy() # binarize the weights

    # inherit the dim2d dictionary of the previous layer(s)
    if skip_connect_layer is not None: # if there is skip connection to the current layer
        # value = (-2,v) means that this connection comes from the skip connection layer
        shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
        prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
    else:
        # value = (-1,v) means that this connection comes from the preceding layer
        merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

    # iterate over each node
    for ri, row in enumerate(Wb):
        # node_activation_cnt is initialized in ProtoRS.detect_dead_node()
        if layer.node_activation_cnt[ri + pos_shift] == 0 or layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot:
            dim2id[ri + pos_shift] = -1 # this probably means that value = -1 marks a dead node 
            continue
        rule = {} # a dictionary of rules for the current node, each rule is a list of node ids from the previous layer having positive connection to this current node
        
        # iterate over each weight of a node, i: node index of previous layers (after concatenation), w: weight of prev node i -> current node ri
        for i, w in enumerate(row):
            if w > 0 and merged_dim2id[i][1] != -1: # previous rule index is not -1 a.k.a valid rule
                rule[merged_dim2id[i]] = 1 # current rule is also valid
        rule = tuple(sorted(rule.keys())) # sort the connection list by previous node's index
        if rule not in rules: # is a unique combination (in the context of this layer)
            rules[rule] = tmp # assign index for rule
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1 # increment unique rules count
        else:
            dim2id[ri + pos_shift] = rules[rule] # reuse index of existing combinations (of this layer)
    return dim2id, rule_list

def dim2idcallable():
    """UnionLayer.dim2id is a defaultdict initialized with a lambda expression. Unfortunately pickling would not work with this lambda expression.
    This is a replacement function for it
    """
    return -1

class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(UnionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)
        self.dis_layer = DisjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat([self.con_layer.binarized_forward(x),
                          self.dis_layer.binarized_forward(x)], dim=1)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        # initialization to prune rule with dead nodes
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = self.node_activation_cnt

        # extract rules from conjunctive and disjunctive layer
        con_dim2id, con_rule_list = extract_rules(prev_layer, skip_connect_layer, self.con_layer)
        dis_dim2id, dis_rule_list = extract_rules(prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[0])

        # concatenate rule ids together
        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        #dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})
        dim2id = defaultdict(dim2idcallable, {**con_dim2id, **dis_dim2id})
        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

        return dim2id, rule_list

    def get_rule_description(self, prev_rule_name, wrap=False):
        self.rule_name = []
        for rl, op in zip(self.rule_list, ('&', '|')): # (conjunctive list, '&'), (disjunctive list, '|')
            # iterate over rules in the list
            for rule in rl:
                name = ''
                # iterate over the rule a.k.a the combination of (concatenated) node IDs from previous layers
                for i, ri in enumerate(rule):
                    op_str = ' {} '.format(op) if i != 0 else ''
                    var_str = ('({})' if wrap else '{}').format(prev_rule_name[2 + ri[0]][ri[1]]) # wrap this rule around previous layer's rules
                    name += op_str + var_str
                self.rule_name.append(name)
