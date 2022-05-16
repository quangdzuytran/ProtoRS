import torch
import torch.nn as nn
import torch.nn.functional as F
from protors.components import Binarize

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
        similarities = self._distances_to_similarities(distances)
        max_similarity = F.max_pool2d(similarities, kernel_size=(W, H))
        # mean_similarity = F.avg_pool2d(similarities, kernel_size=(W, H))
        # focal_similarity = max_similarity - mean_similarity
        # return 
        return max_similarity

    def _l2_convolution(self, xs):
        # Adapted from ProtoPNet
        # Computing ||xs - ps||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
        # where ps is some prototype image

        # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
        # with weights set to 1 so each patch just has its values summed)
        ones = torch.ones_like(self.prototype_vectors,
                               device=xs.device)  # Shape: (num_prototypes, num_features, w_1, h_1)
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


class Binarization(nn.Module):
    def __init__(self, num_prototypes):
        super().__init__()
        self.threshold = nn.Parameter(0.5 * torch.rand(num_prototypes), requires_grad=True)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        binarized = Binarize.apply(xs - self.threshold)
        return binarized


    


