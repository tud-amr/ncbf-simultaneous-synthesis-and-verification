# -*- coding: utf-8 -*-
from dreal import *
import torch 
import numpy as np
import random


def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    ball= Expression(0)
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        ball += x[i]*x[i]
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))
    return CheckSatisfiability(logical_not(condition),config)

def AddCounterexamples(x,CE,N): 
    # Adding CE back to sample set
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb,ub,N)
        nearby.append(nearby_)
    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])             
        x = torch.cat((x, torch.tensor([n_pt])), 0).float()
    return x
  
def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2

def Tune(x):
    # Circle function values
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y

def CheckCBF(x, f, V, config):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    lie_derivative_of_V = Expression(0)
    
    for i in range(len(x)):
        lie_derivative_of_V += f[i]*V.Differentiate(x[i])  
    cbc = lie_derivative_of_V + 0.5*V
    # ball_in_bound = logical_and( - 5* np.pi / 6 <= x[0], 5* np.pi / 6 >= x[0],  -4 <= x[1], 4 >= x[1])
    ball_in_bound = logical_and( - 2 <= x[0], 2 >= x[0],  -4 <= x[1], 4 >= x[1])
    
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_imply(ball_in_bound, cbc >= -1)
    return CheckSatisfiability(logical_not(condition),config)


def CheckCBF2(x, f, V, config):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 
    
    inadmissible_are = logical_not( logical_and( - 5* np.pi / 6 <= x[0], 5* np.pi / 6 >= x[0],  -4 <= x[1], 4 >= x[1]) )
    state_space = logical_and( - np.pi  <= x[0],np.pi >= x[0],  -5 <= x[1], 5 >= x[1] )
    area_check = logical_and(inadmissible_are, state_space)
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_imply(area_check, V <= 0)
    return CheckSatisfiability(logical_not(condition),config)



def AddCounterexamplesCBF(x,CE,N): 
    # Adding CE back to sample set
    c = []
    nearby= []
    
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.random.uniform(lb,ub,N)
        nearby.append(nearby_)

    ce = torch.tensor(c).reshape(1, x.shape[1])

    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])             
        ce = torch.cat((ce, torch.tensor([n_pt])), 0).float()
    
    return ce