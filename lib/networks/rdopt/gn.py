import torch
import torch.nn as nn
import time
from packaging import version
if version.parse(torch.__version__) >= version.parse('1.9'):
    cholesky = torch.linalg.cholesky
else:
    cholesky = torch.cholesky

@torch.jit.script
def computeJtJandJte(x, J, e, lam):
    Jt = torch.transpose(J, dim0=1, dim1=2)
    JtJ = torch.bmm(Jt, J)
    Jte = torch.bmm(Jt, e) 
    return JtJ, Jte



def computeJtJandJte_v2( J , res, weights ): 
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6

 
        grad = weights  * grad
        grad = grad.sum(-2).unsqueeze(-1)  # ... x 6
 
        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6

 
        Hess = weights[..., None] * Hess
        Hess = Hess.sum(-3)  # ... x 6 x6

  
        return Hess,grad


class DampingNet(nn.Module):
    def __init__(self,  num_params=6):
        super().__init__()

        const = torch.zeros(num_params)
        self.register_parameter('const', torch.nn.Parameter(const))


    def forward(self):
        min_  = -6
        max_=5 
        lambda_ = 10.**(min_ + self.const.sigmoid()*(max_ - min_))
 
        return lambda_


def solve_LM(g, H, lambda_=0, mute=False,   eps=1e-6):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
        mask: denotes valid elements of the batch (optional).
    """
    if lambda_ is 0:   
        diag = torch.zeros_like(g)
    else:
        diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
    H = H + diag.clamp(min=eps).diag_embed()
 
    H_, g_ = H, g
    try: 
        U = cholesky(H_) 

    except RuntimeError as e:
        if 'singular U' in str(e):
            if not mute:
                logger.debug(
                    'Cholesky decomposition failed, fallback to LU.')
            delta =  torch.solve(g_[..., None], H_)[0][..., 0]
        else:
            raise
    else:
 
        delta =  torch.cholesky_solve(g_[..., None], U)[..., 0]
 
    return delta 


class GNLayer(nn.Module):
    def __init__(self, out_channels):
        super(GNLayer, self).__init__()
 
    def forward(self, x, e, J,weight,i=None,renta=None):
        bs = x.shape[0]
 
        lambda_ =renta 
        JtJ, Jte = computeJtJandJte_v2(  J, e, weight)
  
        diag = JtJ.diagonal(dim1=-2, dim2=-1) * lambda_ 
        JtJ = JtJ + diag.clamp(min=1e-6).diag_embed() 
 

        try:
 
            delta_x = torch.linalg.solve(JtJ, Jte)[:, :, 0]  
            x = x + delta_x

        except RuntimeError:
            pass

        return x
