import torch
from torch.quantization import get_default_qconfig, prepare, convert

def post_training_static_quantization(model, calibration_loader):
    """
    Performs static post-training quantization on a PyTorch model.
    """
    model.eval()
    model.qconfig = get_default_qconfig('fbgemm')
    prepared_model = prepare(model)
    
    # Calibration step
    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared_model(images)
            
    quantized_model = convert(prepared_model)
    return quantized_model
