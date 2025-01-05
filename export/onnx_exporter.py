import torch

def export_to_onnx(model, dummy_input, output_path='model.onnx'):
    """
    Export PyTorch model to ONNX format with dynamic axes support.
    """
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")
