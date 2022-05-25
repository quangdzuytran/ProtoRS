import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from protors.protors import ProtoRS
from util.log import Log

def project(model: ProtoRS,
            project_loader: DataLoader,
            device,
            args: argparse.Namespace,
            log: Log,  
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    log.log_message("\nProjecting prototypes to most similar training patch (without class restrictions)...")
    # Set the model to evaluation mode
    model.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that maximizes the focal similarity to each prototype
    # To do this we iterate through the train dataset and store for each prototype the most similar latent patch seen so far
    # Also store info about the image that was used for projection
    global_max_proto_sim = {j: -np.inf for j in range(model.num_prototypes)}
    global_max_patches = {j: None for j in range(model.num_prototypes)}
    global_max_info = {j: None for j in range(model.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = model.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and similarities
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - similarities_batch: similarities tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            features_batch, similarities_batch = model.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for j in range(model.num_prototypes):

                # Iterate over all items in the batch
                # Select the features/similarities that are relevant to this prototype
                # - similarities: similarities of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (similarities, patches) in enumerate(zip(similarities_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is most similar to the prototype
                    max_similarity = similarities.max()
                    max_similarity_ix = similarities.argmax()
                    # Use the index to get the most similar latent patch
                    most_similar_patch = patches.view(D, W * H, W1, H1)[:, max_similarity_ix, :, :]

                    # Check if the latent patch is most similar for all data samples seen so far
                    if max_similarity > global_max_proto_sim[j]:
                        global_max_proto_sim[j] = max_similarity
                        global_max_patches[j] = most_similar_patch
                        global_max_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': max_similarity_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'similarity': max_similarity.item(),
                            'most_similar_input': torch.unsqueeze(xs[batch_i],0)
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del similarities_batch
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_max_patches[j].unsqueeze(0) for j in range(model.num_prototypes)),
                                dim=0,
                                out=model.prototype_layer.prototype_vectors.data)
        del projection

    return global_max_info, model