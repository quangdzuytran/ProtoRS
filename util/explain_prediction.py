
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
from subprocess import check_call
import math
from PIL import Image
from protors.protors import ProtoRS
from protors.upsample import find_high_activation_crop, imsave_with_bbox
from protors.components import rule_to_string, get_prototypes_in_rule
import torch

import torchvision
from torchvision.utils import save_image
from collections import defaultdict

def upsample_local(model: ProtoRS,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 img_name: str,
                 prototypes: list,
                 args: argparse.Namespace):
    
    dir = os.path.join(os.path.join(os.path.join(args.log_dir, folder_name),img_name), args.dir_for_saving_images)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        _, similarities_batch = model.forward_partial(sample)
        sim_map = similarities_batch[0,:,:,:].cpu().numpy()
    for prototype_idx in prototypes:
        img = Image.open(sample_dir)
        x_np = np.asarray(img)
        x_np = np.float32(x_np)/ 255
        if x_np.ndim == 2: #convert grayscale to RGB
            x_np = np.stack((x_np,)*3, axis=-1)
        
        img_size = x_np.shape[:2]
        similarity_map = sim_map[prototype_idx]

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map= rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255*rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[...,::-1]
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_latent_similaritymap.png'%str(prototype_idx)), arr=similarity_heatmap, vmin=0.0,vmax=1.0)

        upsampled_act_pattern = cv2.resize(similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_original_image.png'%str(prototype_idx)), arr=overlayed_original_img, vmin=0.0,vmax=1.0)

        # save the highly activated patch
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(similarity_map)] = 0 #mask similarity map such that only the nearest patch z* is visualized
        
        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        plt.imsave(fname=os.path.join(dir,'%s_masked_upsampled_heatmap.png'%str(prototype_idx)), arr=upsampled_prototype_pattern, vmin=0.0,vmax=1.0) 
            
        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.imsave(fname=os.path.join(dir,'%s_nearest_patch_of_image.png'%str(prototype_idx)), arr=high_act_patch, vmin=0.0,vmax=1.0)

        # save the original image with bounding box showing high activation patch
        imsave_with_bbox(fname=os.path.join(dir,'%s_bounding_box_nearest_patch_of_image.png'%str(prototype_idx)),
                            img_rgb=x_np,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

def explain_prediction(model: ProtoRS,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 args: argparse.Namespace,
                 classes: tuple,
                 pred_kwargs: dict = None,
                 ):
    pred_kwargs = pred_kwargs or dict()  # TODO -- assert deterministic routing
    
    print("Generating explanation...")

    # Create dir to store visualization
    img_name = sample_dir.split('/')[-1].split(".")[-2]
    #print(img_name)
    
    if not os.path.exists(os.path.join(args.log_dir, folder_name)):
        os.makedirs(os.path.join(args.log_dir, folder_name))

    destination_folder=os.path.join(os.path.join(args.log_dir, folder_name),img_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    # Get references to where source files are stored
    # upsample_path = os.path.join(os.path.join(args.log_dir,args.dir_for_saving_images),'projected')
    # nodevis_path = os.path.join(args.log_dir,'projected/node_vis')
    # local_upsample_path = os.path.join(destination_folder, args.dir_for_saving_images)

    # Get the model prediction
    with torch.no_grad():
        explain_info = {}
        _, pred  = model.forward(sample, explain_info=explain_info)
        label_ix = torch.argmax(pred, dim=1)[0].item()
        sm = torch.nn.Softmax(dim=1)
        confidence = torch.max(sm(pred),dim=1)[0].item()
        assert 'matched_rules' in explain_info.keys()

    # explained_info['matched_prototypes'] is a list
    # each element in this list is an index for a prototype that exists in a particular image
    matched_prototypes = list(explain_info['matched_prototypes'][0])
    matched_prototypes.sort()

    # explained_info['matched_rules'] is a list
    # each element in this list is a set of matched rules for a particular image    
    matched_rules = explain_info['matched_rules'][0]

    # get the set of prototypes actually used in matched rules.
    prototypes_in_matched_rules = set()
    for rid in matched_rules.keys():
        layer = model.mllp.layer_list[-1 + rid[0]] # layer containing rule
        prototypes_in_matched_rules|= get_prototypes_in_rule(layer.rule_name[rid[1]])
    prototypes_in_matched_rules &= set(matched_prototypes)

    # printing summary
    with open(destination_folder + '/summary.txt','w') as file:
        print('Predicted class: {}'.format(classes[label_ix]),file=file)
        print('Confidence: {:.5f}'.format(confidence), file=file)
        print('Similarity score threshold: {}'.format(explain_info['threshold']), file=file)
        print('Matches {} prototype(s), {} of which result in {} rule(s)'.format(len(matched_prototypes), len(prototypes_in_matched_rules), len(matched_rules)), file=file)
        print('\nMatched prototype(s):',file=file)
        prototype_names = model.prototype_layer.get_prototype_labels()
        for proto_idx in matched_prototypes:
            print('{}'.format(prototype_names[proto_idx]),file=file)

        print('\nPrototype(s) that actually resulted in a matched rule: ', file=file)
        for proto_idx in prototypes_in_matched_rules:
            print('{}'.format(prototype_names[proto_idx]),file=file)

        print('\nMatched rule(s):', file=file)
        for rid in matched_rules.keys():
            print('{}'.format(rid), file=file)
        
    # aggregate rule influence
    Wl = list(model.mllp.layer_list[-1].parameters())[0] # weights and biases of the last layer a.k.a the linear layer
    Wl = Wl.cpu().detach().numpy()
    sig_dict = [] # store the significance of each rule
    for rid in matched_rules.keys():
        significance = 0 # sum of weights for that rule
        for node_idx in matched_rules[rid]:
            significance += Wl[label_ix][node_idx]
        sig_dict.append((significance, rid))
    sig_dict.sort(key=lambda x: abs(x[0]),reverse=True) # sort by significance
    
    # printing rule significance
    with open(destination_folder +'/rule_significance.csv','w') as file:
        print('RID', end=',', file=file)
        print('Significane,Rule', file=file)
        
        for significance, rid in sig_dict:
            print('"{}"'.format(rid), end=',', file=file)
            print('{:.4f}'.format(significance), end=',', file=file)
            now_layer = model.mllp.layer_list[-1 + rid[0]]
            # print('({},{})'.format(now_layer.node_activation_cnt[rid2dim[rid]].item(), now_layer.forward_tot))
            print(rule_to_string(now_layer.rule_name[rid[1]], model.prototype_layer.get_prototype_labels()), end='\n', file=file)
        
    # Save input image
    sample_path = destination_folder + '/sample.jpg'
    # save_image(sample, sample_path)
    Image.open(sample_dir).save(sample_path)

    # Save an image containing the model output
    output_path = destination_folder + '/output.jpg'

    #print("Upsampling locally...")
    upsample_local(model,sample,sample_dir,folder_name,img_name,prototypes_in_matched_rules,args)

    print("Finished")
    """
    # Prediction graph is visualized using Graphviz
    # Build dot string
    s = 'digraph T {margin=0;rankdir=LR\n'
    # s += "subgraph {"
    s += 'node [shape=plaintext, label=""];\n'
    s += 'edge [penwidth="0.5"];\n'

    # Create a node for the sample image
    s += f'sample[image="{sample_path}"];\n'

    # Create nodes for all decisions/branches
    # Starting from the leaf
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()
        
        s += f'node_{i+1}[image="{upsample_path}/{node_ix}_nearest_patch_of_image.png" group="{"g"+str(i)}"];\n' 
        if prob > 0.5:
            s += f'node_{i+1}_original[image="{local_upsample_path}/{node_ix}_bounding_box_nearest_patch_of_image.png" imagescale=width group="{"g"+str(i)}"];\n'  
            label = "Present      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        else:
            s += f'node_{i+1}_original[image="{sample_path}" group="{"g"+str(i)}"];\n'
            label = "Absent      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        # s += f'node_{i+1}_original->node_{i+1} [label="{label}" fontsize=10 fontname=Helvetica];\n'
        
        s += f'node_{i+1}->node_{i+2};\n'
        s += "{rank = same; "f'node_{i+1}_original'+"; "+f'node_{i+1}'+"};"

    # Create a node for the model output
    s += f'node_{len(decision_path)}[imagepos="tc" imagescale=height image="{nodevis_path}/node_{leaf_ix}_vis.jpg" label="{classes[label_ix]}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'

    # Connect the input image to the first decision node
    s += 'sample->node_1;\n'


    s += '}\n'

    with open(os.path.join(destination_folder, 'predvis.dot'), 'w') as f:
        f.write(s)

    from_p = os.path.join(destination_folder, 'predvis.dot')
    to_pdf = os.path.join(destination_folder, 'predvis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s' % (from_p, to_pdf), shell=True)
    """

