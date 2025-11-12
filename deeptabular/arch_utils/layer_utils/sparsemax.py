import torch
from torch.autograd import Function


def _make_ix_like(x, dim=0):
    """
    Creates a tensor of indices like the input tensor along the specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose shape will be used to determine the shape of the output tensor.
    dim : int, optional
        Dimension along which to create the index tensor. Default is 0.

    Returns
    -------
    torch.Tensor
        A tensor containing indices along the specified dimension.
    """
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    Implements the sparsemax function, a sparse alternative to softmax.

    References
    ----------
    Martins, A. F., & Astudillo, R. F. (2016). "From Softmax to Sparsemax: A Sparse Model of
    Attention and Multi-Label Classification."
    """

    @staticmethod
    def forward(ctx, input_, dim=-1):
        """
        Forward pass of sparsemax: a normalizing, sparse transformation.

        Parameters
        ----------
        input_ : torch.Tensor
            The input tensor on which sparsemax will be applied.
        dim : int, optional
            Dimension along which to apply sparsemax. Default is -1.

        Returns
        -------
        torch.Tensor
            A tensor with the same shape as the input, with sparsemax applied.
        """
        ctx.dim = dim
        max_val, _ = input_.max(dim=dim, keepdim=True)
        input_ -= max_val  # Numerical stability trick, as with softmax.
        tau, supp_size = SparsemaxFunction._threshold_and_support(input_, dim=dim)
        output = torch.clamp(input_ - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """
        Backward pass of sparsemax, calculating gradients.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of sparsemax.

        Returns
        -------
        tuple
            Gradients of the loss with respect to the input of sparsemax and None for the dimension argument.
        """
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input_, dim=-1):
        """
        Computes the threshold and support for sparsemax.

        Parameters
        ----------
        input_ : torch.Tensor
            The input tensor on which to compute the threshold and support.
        dim : int, optional
            Dimension along which to compute the threshold and support. Default is -1.

        Returns
        -------
        tuple
            - torch.Tensor : The threshold value for sparsemax.
            - torch.Tensor : The support size tensor.
        """
        input_srt, _ = torch.sort(input_, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input_, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input_.dtype)
        return tau, support_size


def sparsemax(tensor, dim=-1):
    return SparsemaxFunction.apply(tensor, dim)


def sparsemoid(tensor):
    return (0.5 * tensor + 0.5).clamp_(0, 1)
