def get_backward_proxy():
  import torch

  class BackwardProxy(torch.autograd.Function):
    """
    A proxy for Scallop backward
    """

    @staticmethod
    def forward(ctx, input, output, jacobian, sparse_jacobian=False):
      """
      Forward function takes in input x, output y, and jacobian dy/dx
      """
      ctx.save_for_backward(jacobian)
      ctx.sparse_jacobian = sparse_jacobian
      return output

    @staticmethod
    def backward(ctx, grad_output):
      """
      Backward function that propagates the gradient w.r.t. output to
      get the gradient w.r.t. input
      """
      jacobian, = ctx.saved_tensors
      if ctx.sparse_jacobian:
        # An equivalent operation to using einsum under sparse setting
        grad_expanded = grad_output.unsqueeze(-1)
        mult = jacobian * grad_expanded
        mat_f_sparse = torch.sparse.sum(mult, dim=1)
        grad_input = mat_f_sparse.to_dense()
      else:
        grad_input = torch.einsum("ikj,ik->ij", jacobian, grad_output)
      return grad_input, None, None, None

  return BackwardProxy
