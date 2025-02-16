# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:21:54 2025
Function to perform computational complexity analysis
@author: jpeeples
"""

import torch
import torch.nn as nn
import torchinfo
from DSAnet import NMNet
from MSDCNN import MSDCNN

def compute_model_complexity(model, input_size):
    """
    Computes the estimated time complexity (Big-O notation), FLOPs, parameters, and memory usage of a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        input_size (tuple): The input size in the format (batch_size, channels, height, width).

    Returns:
        dict: Dictionary containing model FLOPs, number of parameters, memory usage, and estimated complexity.
    """
    # Get model summary using torchinfo
    model_info = torchinfo.summary(model, input_size=input_size, verbose=0)
    
    total_flops = 0
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = total_params * 4 / (1024 ** 2)  # Memory in MB (assuming float32)

    # Compute total FLOPs per layer
    for layer in model_info.summary_list:
        if hasattr(layer, 'flops'):
            total_flops += layer.flops

    # Estimate Big-O Complexity
    if total_flops < 10**6:
        complexity = "O(n)"
    elif total_flops < 10**9:
        complexity = "O(n^2)"
    elif total_flops < 10**12:
        complexity = "O(n^3)"
    else:
        complexity = "O(n^4) or higher"

    return {
        "Model": model.__class__.__name__,
        "Total Parameters": total_params,
        # "Total FLOPs (M)": total_flops / 1e6,  # Convert FLOPs to millions
        "Total FLOPs": total_flops,
        "Memory Usage (MB)": total_memory,
        "Estimated Complexity": complexity
    }

def compare_models(models, input_size):
    """ Compares multiple models and returns their FLOPs, parameters, and memory usage. """
    comparison_results = []
    
    for model in models:
        model_info = compute_model_complexity(model, input_size)
        comparison_results.append(model_info)
    
    return comparison_results

# Define input tensor size (batch_size, channels, height, width)
input_size = (1, 3, 28, 28)
in_channels = input_size[1]
img_size = input_size[-1]
num_classes = 10

# Example models
DSA = NMNet(num_channels=in_channels, img_size=img_size, num_classes=num_classes)
MSDC = MSDCNN(num_channels=in_channels, img_size=img_size, num_classes=num_classes)

# Create models
models = [DSA, MSDC]

# Compare models
comparison_results = compare_models(models, input_size)

# Display comparison results
for result in comparison_results:
    print(f"Model: {result['Model']}")
    print(f"  - Total Parameters: {result['Total Parameters']}")
    # print(f"  - FLOPs (M): {result['Total FLOPs (M)']:.2f}")
    print(f"  - FLOPs: {result['Total FLOPs']:.2f}")
    print(f"  - Memory Usage (MB): {result['Memory Usage (MB)']:.2f}")
    print(f"  - Estimated Complexity: {result['Estimated Complexity']}")
    print("-" * 40)