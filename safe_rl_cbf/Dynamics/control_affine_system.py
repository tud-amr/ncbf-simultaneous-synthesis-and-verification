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

    def __init__(self, ns=2, nu=1, nd=1, dt=0.01):
        super().__init__()
        
        assert ns > 0 and nu >= 0 and nd >= 0 and dt > 0
        self.ns = ns
        self.nu = nu
        self.nd = nd
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

    @abstractmethod
    def d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the disturbance-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state

        returns:
            d: bs x self.n_dims x self.n_disturbances tensor
        """
        pass

    
    @abstractmethod
    def range_dxdt(self, x_range: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the range of dsdt(x,u) for all s in the batch.

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
            u: a tensor of (batch_size, self.n_controls) points in the control space
        returns:
            a tuple (lower, upper) of tensors of (batch_size, self.n_dims) points
            giving the lower and upper bounds on dxdt(x,u) for all x in the batch.
        """
        pass


    def set_domain_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):
        
        message = f"lower_bd and upper_bd must the same shape but got {lower_bd.shape} and {upper_bd.shape}"
        assert lower_bd.shape == upper_bd.shape, message
        message = f"lower_bd and upper_bd must be tensors of shape ({self.ns,},)"
        assert lower_bd.shape[0] == self.ns, message

        self.domain_lower_bd = lower_bd
        self.domain_upper_bd = upper_bd
    
    def set_control_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):

        message = f"lower_bd and upper_bd must the same shape but got {lower_bd.shape} and {upper_bd.shape}"
        assert lower_bd.shape == upper_bd.shape, message
        message = f"lower_bd and upper_bd must be tensors of shape ({self.nu}, )"
        assert lower_bd.shape[0] == self.nu, message

        self.control_lower_bd = lower_bd
        self.control_upper_bd = upper_bd

    def set_disturbance_limits(self, lower_bd: torch.Tensor,upper_bd: torch.Tensor):

        message = f"lower_bd and upper_bd must the same shape but got {lower_bd.shape} and {upper_bd.shape}"
        assert lower_bd.shape == upper_bd.shape, message
        message = f"lower_bd and upper_bd must be tensors of shape ({self.nd}, )"
        assert lower_bd.shape[0] == self.nd, message

        self.disturbance_lower_bd = lower_bd
        self.disturbance_upper_bd = upper_bd


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

    def dsdt(self, s: torch.Tensor, u: torch.Tensor, d : torch.Tensor = None) -> torch.Tensor:
        """
        Return the state derivative dsdt for a given state and control input.
        s: (bs, ns)
        u: (bs, nu)
        d: (bs, nd)
        """

        ds = self.f(s) + torch.bmm(self.g(s), u.unsqueeze(dim=-1)).squeeze(dim=-1)
    
        return ds

    def step(self, s: torch.Tensor, u: torch.Tensor, dt=None) -> torch.Tensor:
        ds = self.dsdt(s, u)
        if dt is None:
            s_next = s + ds * self.dt
        else:
            s_next = s + ds * dt
        return s_next

    def normalize_angle(self, theta):
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    @property
    def domain_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of states for this system
        upper : (ns,)
        lower : (ns,)
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.domain_upper_bd
        lower_limit = self.domain_lower_bd

        return (lower_limit, upper_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple ( upper: (nu,), lower: (nu,) ) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.control_upper_bd
        lower_limit = self.control_lower_bd

        return (lower_limit, upper_limit)
    
    @property
    def disturbance_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple ( upper: (nd,), lower: (nd,) ) describing the range of allowable disturbance
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = self.disturbance_upper_bd
        lower_limit = self.disturbance_lower_bd

        return (lower_limit, upper_limit)

    @property
    def K(self) -> torch.Tensor:
        """
        Return a tensor K, which is gain from LQR controller
        """
        # define upper and lower limits based around the nominal equilibrium input
        
        return self.K_lqr

    