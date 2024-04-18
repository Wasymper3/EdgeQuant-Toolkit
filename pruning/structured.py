import torch
import torch.nn.utils.prune as prune

def prune_conv_layers(model, amount=0.3):
    """
    Apply structured L1-norm pruning to convolutional layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')
    return model
