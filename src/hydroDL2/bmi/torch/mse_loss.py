import torch
from torch.autograd import Function

class CustomMSELoss(Function):
    """
    Compute gradient for MSE loss function:

        L := MSE = sum((y - y*)^2)

        dL/dw = del(L)/del(y) * dy/dw

        del(L)/del(y) = 2 * (y_pred - y_obs)

    Custom autograd with torch.autograd.Function for BMI-friendly backward pass.

    Essentially, we need this to calculate dL/dw = del(L)/del(y) * dy/dw,
    then torch Autodiff system can do the rest (e.g., calc vector Jacobian prod).
    """
    @staticmethod
    def forward(ctx, y_pred, y_true):
        """
        Compute MSE loss and save prediction and observation 
        tensors for backward pass.

        Receive a tensor y_pred and y_true, and stash in the context object ctx for the backward.

        :param ctx: context object to save tensors for backward pass
        :param y_pred: predicted values
        :param y_true: true values
        """
        # Save for backward
        ctx.save_for_backward(y_pred, y_true)
        
        # Compute MSE loss
        loss = torch.mean((y_pred - y_true) ** 2)

        return loss

    @staticmethod
    def backward(ctx, internal_grad):
        """
        Compute the gradient of the loss with respect to y_pred.

        :param ctx: Context object to save tensors for backward pass
        :param internal_grad: Gradients from further down computational graph
            (i.e., dy/dw, or anything coming from before y_pred & y_obs tensor
            reconstruction)
        """
        # Retrieve saved tensors
        y_pred, y_true = ctx.saved_tensors
        
        # Number of elements in the batch
        n = y_pred.size(0)
        
        # Grad of the loss with respect to y_pred.
        grad_y_pred = 2 * (y_pred - y_true) / n
        
        # incorporate internal_grad to account for additional grads from
        # further down the graph.
        grad_y_pred = grad_y_pred * internal_grad
        
        # Return the gradient for y_pred, and no grad w.r.t. y_true.
        return grad_y_pred, None


if __name__ == '__main__':
    # Test the custom MSE loss
    y_pred = torch.tensor([2.5, 0.0, 2.0], requires_grad=True)
    y_true = torch.tensor([1.0, 0.0, -1.0])

    # Use the custom MSE loss
    loss = CustomMSELoss.apply(y_pred, y_true)

    # Perform backward pass
    loss.backward()

    # Check gradients
    print(y_pred.grad)  # This will print the gradient for y_pred
