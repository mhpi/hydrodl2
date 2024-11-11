"""
A custom backward method to enable gradient backpropogation in a
physics-informed, differentiable ML BMI.
"""
import torch
import numpy as np
from typing import Optional
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge

import bmi_dpl_model as bmi_model



class BMIBackward(torch.autograd.Function):
    """
    Custom autograd with torch.autograd.Function for BMI-friendly backward pass 
    implementation which operate on Tensors.

    Essentially, we need this to calculate dL/dw = del(L)/del(y) * dy/dw,
    then torch Autodiff system can do the rest (e.g., calc vector Jacobian prod).
    """
    @staticmethod
    def forward(self, ctx, input_tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output.
        ctx is a context object that can be used to stash information for
        backward computation. 
        You can cache arbitrary objects for use in the backward pass using
        the ctx.save_for_backward method.
        """
        # Convert PyTorch tensor to NumPy array
        input_np = input_tensor.detach().cpu().numpy()

        # Call the BMI forward (.update())
        output_np = bmi_model.update(input_np)

        # Save the input for backward pass
        ctx.save_for_backward(input_tensor)

        # Convert output back to a PyTorch tensor
        output_tensor = torch.from_numpy(output_np).to(input_tensor.device)

        return output_tensor  #.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """
        # Retrieve the saved tensor from forward pass
        input_tensor, = ctx.saved_tensors

        # Convert PyTorch tensor to NumPy array for BMI gradient computation
        grad_output_np = grad_output.detach().cpu().numpy()

        # Compute gradients using BMI model
        grad_input_np = BMIBackward.compute_bmi_gradient(input_tensor.cpu().numpy(), grad_output_np)

        # Convert gradients back to PyTorch tensor
        grad_input_tensor = torch.from_numpy(grad_input_np).to(input_tensor.device)

        return grad_input_tensor, None  # No gradient w.r.t. other inputs

    
    def compute_bmi_gradient(y_pred, grad_output_np):
        """
        Compute gradient for MSE loss function:
        
            L := MSE = sum((y - y*)^2)

            dL/dw = del(L)/del(y) * dy/dw

            del(L)/del(y) = 2 * (y_pred - y_obs)
        """
        # Here we are computing the MSE gradient, but adapt as needed for the BMI model
        mse_gradient = 2 * (y_pred - grad_output_np)  # y_pred is the input, grad_output_np is the gradient passed from previous layers
        return mse_gradient

    @staticmethod
    def backward(self,
                 ctx: _TensorOrTensors,
                 grad_tensors: Optional[_TensorOrTensors] = None,
                #  retain_graph: Optional[bool] = None,
                #  create_graph: bool = False,
                #  grad_variables: Optional[_TensorOrTensors] = None,
                 inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
                 ) -> None:
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # Retrieve the saved tensor from forward pass
        input_tensor, = ctx.saved_tensors

        # Convert PyTorch tensor to NumPy array for BMI gradient computation
        grad_output_np = grad_tensors.detach().cpu().numpy()

        # Compute gradients using BMI model (replace with actual gradient call)
        grad_input_np = self.compute_bmi_gradient(input_tensor.cpu().numpy(), grad_output_np)

        # Convert gradients back to PyTorch tensor
        grad_input_tensor = torch.from_numpy(grad_input_np).to(input_tensor.device)
    
        # input, = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # return grad_input

        # return grad_input_tensor
    