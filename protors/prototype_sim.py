import torch
import torch.nn as nn
import torch.nn.functional as F
from protors.components import Binarize

class Similarity(nn.Module):
    def __init__(self, 
                 num_prototypes: int, 
                 num_features: int, 
                 w_1: int, 
                 h_1: int,
                 epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        self.num_features = num_features
        prototypes_shape = (num_prototypes, num_features, w_1, h_1)
        self.prototype_vectors = nn.Parameter(torch.rand(prototypes_shape), requires_grad=True)

    def forward(self, xs: torch.Tensor, W: int, H: int) -> torch.Tensor:
        distances = self._l2_convolution(xs)
        similarities = self._distances_to_similarities(distances)
        max_similarity = F.max_pool2d(similarities, kernel_size=(W, H))
        # mean_similarity = F.avg_pool2d(similarities, kernel_size=(W, H))
        # focal_similarity = max_similarity - mean_similarity
        # return focal_similarity
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
        # distances = torch.sqrt(torch.abs(distances) + self.epsilon)

        return distances

    def _distances_to_similarities(self, distances):
        # return torch.log((distances + 1) / (distances + self.epsilon))
        # return 1 / (1 + distances + self.epsilon)
        w_1 = self.prototype_vectors.shape[-2]
        h_1 = self.prototype_vectors.shape[-1]
        similarities = 1 - torch.sqrt(distances / torch.tensor(self.num_features * w_1 * h_1) + self.epsilon)
        return similarities.clamp(0, 1)
    
    def get_prototype_labels(self):
        num_prototypes = self.prototype_vectors.shape[0]
        padding_width = len(str(num_prototypes))
        return ['p_'+ str(i).rjust(padding_width,'0') for i in range(num_prototypes)]


class Binarization(nn.Module):
    def __init__(self, num_prototypes, binarize_threshold):
        super().__init__()
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(num_prototypes)}
        self.threshold = binarize_threshold
        self.k = 50
        self.hard_threshold = False

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.k * (xs - self.threshold))

    def binarized_forward(self, 
                        xs: torch.Tensor,
                        explain_info: dict = None) -> torch.Tensor:
        with torch.no_grad():
            # extra code for local explanation 
            if explain_info is not None:
                explain_info['threshold'] = self.threshold
                # record which prototype is a match
                binarized = Binarize.apply(xs - self.threshold)
                matched_prop_idx = binarized.nonzero().to('cpu').numpy() # non-zeroes are matches

                # turn nonzero()'s output format into our own format
                matched_prop_list = [set() for x in xs] # a list of matched prototypes set for every sample
                for indices in matched_prop_idx:
                    matched_prop_list[indices[0]].add(self.dim2id[indices[1]]) # indices[0]: index of sample in the batch

                explain_info['matched_prototypes'] = matched_prop_list

            # MAIN forward code
            if self.hard_threshold:
                return Binarize.apply(xs - self.threshold)
            else:
                return self.forward(xs)

    


