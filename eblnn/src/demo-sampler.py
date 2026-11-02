import torch
from typing import Callable

class LangevinSampler:
    """
    Stripped-down Langevin dynamics sampler.
    """
    #first, we initialize our variables
    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        n_steps: int = 50,
        alpha: float = 0.01, 
    ) -> None:
        self.energy_fn = energy_fn
        self.n_steps = n_steps
        self.alpha = alpha

    #now the sampling procedure
    def sample(self, x_init: torch.Tensor) -> torch.Tensor:
        """
        Runs the SDE discretization:
        x_{t+1} = x_t - (alpha / 2) * ∇E(x_t) + sqrt(alpha) * z_t
        """
        x = x_init.clone().detach()
        x.requires_grad_(True)
        
        
        trajectory = [x.detach().clone()]

        for _ in range(self.n_steps):
            #1. calculate energy
            energy = self.energy_fn(x)
            
            #calculate gradient
            if x.grad is not None:
                x.grad.zero_()
            energy.sum().backward()
            grad = x.grad.detach() 

            #calculate how much we drift deterministically
            drift = (self.alpha / 2) * grad
            
            # calculathe the random noise
            noise = torch.sqrt(torch.tensor(self.alpha)) * torch.randn_like(x)
            
            # Update position
            x_new = x.detach() - drift + noise

            # Prepare for next discrete time step
            x = x_new.detach()
            x.requires_grad_(True)
            
            trajectory.append(x.detach().clone())

        return x.detach(), trajectory