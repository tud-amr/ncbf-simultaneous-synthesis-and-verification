from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
import torch
from typing import Tuple, Optional, Union


class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamical system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + g(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(self, ns=2, nu=1, dt=0.01):
        super().__init__()
        
        assert ns > 0 and nu >= 0 and dt > 0
        self.ns = ns
        self.nu = nu
        self.dt = dt


    @abstractmethod
    def f(self, s: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        pass
    

    @abstractmethod
    def g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        pass

    def set_domain_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):
        self.domain_lower_bd = lower_bd
        self.domain_upper_bd = upper_bd
    
    def set_control_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):
        self.control_lower_bd = lower_bd
        self.control_upper_bd = upper_bd

    def set_state_constraints(self, rou):
        """
        rou(s) is a function such that rous(s) >= 0 implies system is safe.
        """
        self.state_constraints = rou

    def set_nominal_state_constraints(self, rou_n):
        '''
        rou_n(s) is a function that ros_n(s) >= 0 implies system is nominal safe. Usually this safe set is 
        hand-designed and conservative.
        '''
        self.nominal_state_constraints = rou_n

    def set_barrier_function(self, h):
        self.h = h


    def safe_mask(self, s: torch.Tensor) -> Tuple[bool]:
        rou_s = self.nominal_state_constraints(s)
        x = rou_s >=0
        safe_mask = torch.all(x, dim=1)
        return safe_mask

    def unsafe_mask(self, s:torch.Tensor) -> Tuple[bool]:
        rou_s = self.state_constraints(s)
        x= rou_s >= 0
        unsafe_mask = torch.all(x, dim=1)
        return ~unsafe_mask

    def boundary_mask(self, s: torch.Tensor) -> torch.Tensor:
        """Return the mask of s indicating regions that are neither nominal_safe nor unsafe

        args:
            s: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return torch.logical_not(
            torch.logical_or(
                self.safe_mask(s),
                self.unsafe_mask(s),
            )
        )

    def dsdt(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ds = self.f(s) + torch.bmm(self.g(s), u.unsqueeze(dim=-1)).squeeze(dim=-1)

        return ds

    def step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ds = self.dsdt(s, u)
        s_next = s + ds * self.dt
        return s_next


    @property
    def domain_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of states for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.domain_upper_bd
        lower_limit = self.domain_lower_bd

        return (lower_limit, upper_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.control_upper_bd
        lower_limit = self.control_lower_bd

        return (lower_limit, upper_limit)
    
    @property
    def K(self) -> torch.Tensor:
        """
        Return a tensor K, which is gain from LQR controller
        """
        # define upper and lower limits based around the nominal equilibrium input
        
        return self.K_lqr

    