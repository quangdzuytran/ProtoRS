import torch
import torch.nn as nn
import torch.nn.functional as F
from protors.components import Binarize, EPSILON
import math

class FocalSimilarity(nn.Module):
    def __init__(self, 
                 num_prototypes: int, 
                 num_features: int, 
                 w_1: int, 
                 h_1: int,
                 epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        prototypes_shape = (num_prototypes, num_features, w_1, h_1)
        self.prototype_vectors = nn.Parameter(torch.randn(prototypes_shape), requires_grad=True)

    def forward(self, xs: torch.Tensor, W: int, H: int) -> torch.Tensor:
        distances = self._l2_convolution(xs)

        #FIXME: below lines were commented to test DSQ
        #similarities = self._distances_to_similarities(distances) 
        #max_similarity = F.max_pool2d(similarities, kernel_size=(W, H))
        # mean_similarity = F.avg_pool2d(similarities, kernel_size=(W, H))
        # focal_similarity = max_similarity - mean_similarity
        # return focal_similarity
        #return max_similarity

        #FIXME: below lines are for testing DSQ. Return in L2 distance unit
        #max_distance = F.max_pool2d(distances, kernel_size=(W, H))
        min_distance = -F.max_pool2d(-distances, kernel_size=(W, H))
        return (256-min_distance)/256 #FIXME:scaled to [0,1]
        

    def _l2_convolution(self, xs):
        # Adapted from ProtoPNet
        # Computing ||xs - ps||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
        # where ps is some prototype image

        # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
        # with weights set to 1 so each patch just has its values summed)
        ones = torch.ones_like(self.prototype_vectors, device=xs.device)  # Shape: (num_prototypes, num_features, w_1, h_1)
        xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)  # Shape: (bs, num_prototypes, w_in, h_in)

        # Now compute ||ps||^2
        # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
        # squared L2 distance
        ps_squared_l2 = torch.sum(self.prototype_vectors ** 2,
                                  dim=(1, 2, 3))  # Shape: (num_prototypes,)
        # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)

        # Compute xs * ps (for all patches in the input image)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)  # Shape: (bs, num_prototypes, w_in, h_in)

        # Use the values to compute the squared L2 distance
        distances = F.relu(xs_squared_l2 + ps_squared_l2 - 2 * xs_conv)

        return distances

    def _distances_to_similarities(self, distances):
        # return torch.log((distances + 1) / (distances + self.epsilon))
        return 1 / (1 + distances + self.epsilon)

class STEFunction(torch.autograd.Function):
    """Sign function with Straight Through Estimator as gradient"""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), -torch.ones_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        #print(torch.min(grad_input).item(), torch.max(grad_input).item())
        #return grad_input
        return F.hardtanh(grad_input) # gradient clipping

class Binarization(nn.Module):
    def __init__(self, num_prototypes):
        super().__init__()
        #self.threshold = nn.Parameter(0.5 * torch.rand(num_prototypes), requires_grad=True)
        #self.threshold = nn.Parameter(data=torch.Tensor(0.9).float(), requires_grad=False)
        #print(self.threshold)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        #print(self.threshold.item())
        #binarized = Binarize.apply(xs - self.threshold)
        #binarized = Binarize.apply(xs - 0.9)
        sign = STEFunction.apply(xs - 0.9)
        #return sign
        binarized = (sign + 1) * 0.5
        return binarized        
        
        #FIXME: test if this layer is causing model not to diverge. Edit: it is
        #return xs

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round() * 2 - 1
    @staticmethod
    def backward(ctx, g):
        return g

class DSQ(nn.Module):
    """ A quick and dirty adaptation of Differentiable Soft Quantization (DSQ) to ProtoRS. 
    Not intended to be transfered to other models
    """
    def __init__(self, max_value, num_prototypes):
        super().__init__()
        self.threshold = nn.Parameter(data=torch.tensor(0.9).float(), requires_grad=False)
        self.l = nn.Parameter(data=torch.tensor(0).float(), requires_grad=False)
        self.u = nn.Parameter(data=torch.tensor(1).float(), requires_grad=False)
        self.alpha = nn.Parameter(data=torch.tensor(0.2).float(), requires_grad=True)
    
    def delta(self):
        return self.u - self.l

    def _clip(self, x, lower_bound, upper_bound):
        # clip lower
        x = x + F.relu(lower_bound - x)
        # clip upper
        x = x - F.relu(x - upper_bound)
        return x

    def phi_function(self, x, mi, delta, alpha):
        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        #alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1 / (1 - alpha)
        k = (2/ alpha - 1).log() * (1/delta)
        x = (((x - mi) * k ).tanh()) * s 
        return x	

    def dequantize(self, x, lower_bound, delta):

        # save mem
        x =  ((x+1)/2) * delta + lower_bound

        return x

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # 1. Clip input using u and l
        #xs = (self.u - xs)/self.delta()
        #print(torch.min(xs).item(), torch.max(xs).item())
        #if (torch.min(xs) < self.l or torch.max(xs) > self.u):
            #print("\nOut of bound value detected! min = {0}, max = {1}".format(torch.min(xs), torch.max(xs)))
        #xs = (self.u - xs)/self.delta()
        #return xs
        #print("")
        #xs = xs - self.threshold
        #print("")
        #print(self.l.item(), self.u.item(), self.alpha.item(), self.threshold.item())
        #print((self.u.item() + self.l.item()) / 2, self.alpha.item())
        self._clip(xs, self.l, self.u)

        # 2. Apply the Phi function
        delta = self.delta()
        alpha = self.alpha + EPSILON
        mi = self.l + 0.5 * delta
        s = 1 / (1 - alpha)
        k = (2/ alpha - 1).log() * (1/delta)
        xs = (((xs - mi) * k ).tanh()) * s
        #print(torch.min(xs), torch.max(xs))

        # 3. Keep consistent with standard binarization

        deltaX = s*2
        xs = (xs+s)/deltaX
        #xs = (xs - self.l)/delta
        xs = RoundWithGradient.apply(xs)
        #xs = Binarize.apply(xs)
        #print(torch.min(xs), torch.max(xs))

        # 4. Dequantization - not necessary
        xs = self.dequantize(xs, self.l, delta)
        #xs = (xs + 1) * 0.5 # round from -1 or 1 to 0 or 1
        #print(torch.min(xs), torch.max(xs))

        return xs