from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.DataModule import DataModule
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem


######################### define neural network #########################

class NeuralCBF(pl.LightningModule):
    def __init__(
        self,
        dynamic_system: ControlAffineSystem,
        train_mode: int = 0,
        primal_learning_rate: float = 1e-3,
        gamma: float = 0.9,
        clf_lambda: float = 0.5,
        ):

        super(NeuralCBF, self).__init__()
        self.dynamic_system = dynamic_system
        self.train_mode = train_mode
        self.dt = self.dynamic_system.dt
        self.gamma = gamma
        self.clf_lambda = clf_lambda
        self.use_h0 = False

        # initialize learning rate
        if train_mode == 0:
            self.learning_rate = primal_learning_rate
        elif (train_mode == 1 or train_mode == 3):
            self.learning_rate = 5e-4
        elif train_mode == 2:
            self.learning_rate = 5e-5

        
        # initialize model
        # self.h = nn.Sequential(
        #         nn.Linear(self.dynamic_system.ns, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 1)
        #     )

        self.h = nn.Sequential(
                nn.Linear(self.dynamic_system.ns, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        
        self.param_init(self.h)

        self.h0 = copy.deepcopy(self.h)
        
        # generate control vertices
        self.u_v = self.dynamic_system.control_vertices
        self.d_v = self.dynamic_system.disturb_vertices
        
    
        self.descent_loss_name = ['descent_term_linearized', 'descent_term_simulated', 'hji_vi_descent_loss_term']
        self.safety_loss_name = ['unsafe_region_term','unsafe_area_hji_term', 'regulation_loss_term', 'below_violation']
        self.performance_lose_name = ['safe_region_term', 'descent_boundary_term', 'hji_vi_boundary_loss_term']

        self.coefficient_for_performance_loss = 1
        self.coefficient_for_inadmissible_state_loss = 5
        self.coefficients_for_descent_loss = 5

        self.hji_vi_boundary_loss_term = torch.tensor([0.0])
        self.hji_vi_descent_loss_term = torch.tensor([0.0])

    @staticmethod
    def param_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform(m.weight) 

    # def prepare_data(self):
    #     return self.data_module.prepare_data()

    # def setup(self, stage: Optional[str] = None):
    #     return self.data_module.setup(stage)

    # def train_dataloader(self):
    #     # self.data_module.set_dataset()
    #     return self.data_module.train_dataloader()

    # def val_dataloader(self):
    #     return self.data_module.val_dataloader()

    # def test_dataloader(self):
    #     return self.data_module.test_dataloader()

    def set_previous_cbf(self, h):
        self.h0.load_state_dict(h.state_dict())
        self.use_h0 = True
        
    def forward(self, s):
        hs = self.h(s)

        return hs

    def jacobian(self, y, x):
        ''' jacobian of y wrt x '''
        # meta_batch_size, num_observations = y.shape[:2]
        # jac = torch.zeros(meta_batch_size, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
        # for i in range(y.shape[-1]):
        #     # calculate dydx over batches for each feature value of y
        #     y_flat = y[...,i].view(-1, 1)
        #     jac[:, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

        # status = 0
        # if torch.any(torch.isnan(jac)):
        #     status = -1
        x_norm = x 

        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.dynamic_system.ns, self.dynamic_system.ns)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamic_system.ns):
            JV[:, dim, dim] = 1.0

        # Now step through each layer in V
        V = x_norm
        for layer in self.h:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

       
        return JV

    def V_lie_derivatives(
        self, 
        x: torch.Tensor,
        gradh: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """

        # Get the Jacobian of V for each entry in the batch
        # _, gradh = self.V_with_jacobian(x)
        # print(f"gradh shape is {gradh.shape}")
        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, 2, 1)
        Lg_V = torch.zeros(batch_size, 2, 1)

        f = self.dynamic_system.f(x)
        g = self.dynamic_system.g(x)
        d = self.dynamic_system.d(x)
        # print(f"f shape is {f.shape}")
        # print(f"g shape is {g.shape}")

        Lf_V = torch.bmm(gradh, f.unsqueeze(dim=-1)).squeeze(1)
        Lg_V = torch.bmm(gradh, g).squeeze(1)
        Ld_V = torch.bmm(gradh, d).squeeze(1)
        # print(f"Lf_V shape is {Lf_V.shape}")
        # print(f"Lg_V shape is {Lg_V.shape}")
        
        return Lf_V, Lg_V, Ld_V

    def epsilon_area_loss(self,
        s : torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        grid_gap: torch.Tensor,
        model: BoundedModule,
        coefficients_unsafe_epsilon_loss: float=1e2,
    ) -> List[Tuple[str, torch.Tensor]]:
        '''
            compute the epsilon area loss to ensure the continuous satisfaction.
        
        '''
        loss = []
        gridding_gap = grid_gap
        data_l = s - gridding_gap
        data_u = s + gridding_gap
        
        # define perturbation
        # perturbation = PerturbationLpNorm(norm=np.inf, x_L=data_l, x_U=data_u)
        # define perturbed data
        # x = BoundedTensor(s, perturbation)

        # lb, ub = model.compute_bounds(x=(x,), method="backward")
        
        # compute lb and ub by sampling
        N_samples = 50
        lb_list = []
        ub_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s) * (data_u - data_l) + data_l
            h_sample = self.h(x_sample)
            lb_list.append(h_sample)
            ub_list.append(h_sample)
        
        lb = torch.stack(lb_list, dim=0).min(dim=0)[0]
        ub = torch.stack(ub_list, dim=0).max(dim=0)[0]
        
        indices = torch.where(unsafe_mask>0)

        for i in indices[0]:
            
            # augment the data
            data = (s[i].unsqueeze(dim=0), grid_gap[i].unsqueeze(dim=0))
            node = self.data_module.new_tree.get_node(self.data_module.uniname_of_data(data))
            if ub[i] > 0:
                node.data[2] = False
            else:
                node.data[2] = True
            
            

        # hs_unsafe_ub = ub[unsafe_mask]
        # unsafe_ub_violation = coefficients_unsafe_epsilon_loss * F.relu(hs_unsafe_ub)

        # unsafe_hs_epsilon_term = unsafe_ub_violation.mean()
        # if not torch.isnan(unsafe_hs_epsilon_term):
        #     loss.append(("unsafe_region_epsilon_term", unsafe_hs_epsilon_term))

        # print(f"unsafe_region_epsilon_term = {unsafe_hs_epsilon_term}")
        
        return loss 

    def epsilon_decent_loss(
        self,
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        grid_gap: torch.Tensor,
        hs: torch.Tensor,
        gradh: torch.Tensor,
        model: BoundedModule,
        model_jacobian: BoundedModule,
        coefficients_descent_loss: float=1e2,
        accuracy: bool = False,
    )  -> List[Tuple[str, torch.Tensor]]:

        ################################# compute bound on epsilon area #####################################
        
        loss = []

        positive_mask =  torch.logical_not(unsafe_mask) # torch.logical_and(torch.logical_not(unsafe_mask), hs < 1)
        
        ############################ compute epslon area decrease loss #########################################

        bs = s.shape[0]
        gridding_gap = grid_gap/2
        data_l = s - gridding_gap
        data_u = s + gridding_gap

        
        # number_of_samples = 50
        # dxdt_list = []
        # for i in range(number_of_samples):
        #     sample = (data_u - data_l) * torch.rand_like(data_l) + data_l
        #     u, d_min = self.get_control_disturb_vertices(sample)
        #     # print(f"sample  is {sample}")
        #     try:
        #         dxdt = self.dynamic_system.dsdt(sample, u=u)
        #     except:
        #         print(f"u is {u}")
        #     dxdt_list.append(dxdt)

        # dxdt = torch.stack(dxdt_list, dim=0)

        # dxdt_min, _ = torch.min(dxdt, dim=0)
        # dxdt_max, _ = torch.max(dxdt, dim=0)
      

        u_qp, d_min = self.get_control_disturb_vertices(s)


        lb_dx, ub_dx = self.dynamic_system.range_dxdt(data_l, data_u, u_qp)
        # lb_dx = dxdt_min
        # ub_dx = dxdt_max
        # print(f"lb_dx is {lb_dx}")
        # print(f"ub_dx is {ub_dx}")


        # # define perturbation
        # perturbation = PerturbationLpNorm(norm=np.inf, x_L=data_l, x_U=data_u)
        # # define perturbed data
        # x = BoundedTensor(s, perturbation)
        
        # #### A lower upper 
        # required_A = defaultdict(set)
        # required_A[model.output_name[0]].add(model.input_name[0])

        # lb_h, ub_h = model.compute_bounds(x=(x,), method="backward")

        # compute lb and ub by sampling
        N_samples = 50
        lb_list = []
        ub_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s)* (data_u - data_l) + data_l
            h_sample = self.h(x_sample)
            lb_list.append(h_sample)
            ub_list.append(h_sample)
        
        lb_h = torch.stack(lb_list, dim=0).min(dim=0)[0]
        ub_h = torch.stack(ub_list, dim=0).max(dim=0)[0]

        # print(f"lb is {lb}")
        # print(f"ub is {ub}")
        # A_lower, b_lower = A_dict[model.output_name[0]][model.input_name[0]]['lA'], A_dict[model.output_name[0]][model.input_name[0]]['lbias']
        # A_upper, b_upper = A_dict[model.output_name[0]][model.input_name[0]]['uA'], A_dict[model.output_name[0]][model.input_name[0]]['ubias']
        
        
        ########### A_J_lower upper

        # jacobian_lower, jacobian_upper = model_jacobian.compute_jacobian_bounds(x)

        # compute jacobian bound by sampling
        N_samples = 50
        jacobian_lower_list = []
        jacobian_upper_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s) * (data_u - data_l) + data_l
            hs_sample = self.h(x_sample)
            jacobian_sample = self.jacobian(hs_sample, x_sample)

            jacobian_lower_list.append(jacobian_sample)
            jacobian_upper_list.append(jacobian_sample)

        jacobian_lower = torch.stack(jacobian_lower_list, dim=0).min(dim=0)[0]
        jacobian_upper = torch.stack(jacobian_upper_list, dim=0).max(dim=0)[0]

        # print(f"lower_jacobian is {jacobian_lower}")
        # print(f"upper_jacobian is {jacobian_upper}")
        lb_j = jacobian_lower.reshape(bs, -1).to(s.device)
        ub_j = jacobian_upper.reshape(bs, -1).to(s.device)
        # print(f"lb_j is {lb_j}")
        # print(f"ub_j is {ub_j}")

        # compute the multiplication of lower bound and upper bound
        lb_j_lb_dx = torch.mul(lb_j, lb_dx)
        ub_j_ub_dx = torch.mul(ub_j, ub_dx)
        lb_j_ub_dx = torch.mul(lb_j, ub_dx)
        ub_j_lb_dx = torch.mul(ub_j, lb_dx)

        # sovlve convex relaxation problem
        # J, dx, X, h= self._solve_convex_relataxion(lb_j, ub_j, lb_dx, ub_dx, lb_h, ub_h, lb_j_lb_dx, ub_j_ub_dx, lb_j_ub_dx, ub_j_lb_dx)

        # q_min = torch.sum(X, dim=1, keepdim=True) + self.clf_lambda * h

        q_min_1 = torch.sum(lb_j_lb_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_2 = torch.sum(ub_j_ub_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_3 = torch.sum(lb_j_ub_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_4 = torch.sum(ub_j_lb_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h

        q_min = torch.min(torch.min(q_min_1, q_min_2), torch.min(q_min_3, q_min_4))
        
        indices = torch.where(positive_mask.squeeze(dim=-1)>0)
       

        for i in indices[0]:
            # augment the data
            data = (s[i].unsqueeze(dim=0), grid_gap[i].unsqueeze(dim=0))
            node = self.data_module.new_tree.get_node(self.data_module.uniname_of_data(data))
            if q_min[i] < 0.0:
                node.data[2] = False
            else:
                node.data[2] = True
          

        return loss


    def verify(self,
                s : torch.Tensor,
                grid_gap: torch.Tensor,
                safe_mask: torch.Tensor,
                unsafe_mask: torch.Tensor,
                satisfied: torch.Tensor,
                ):
        # chech if unsafe hyperrectangles are all below zero
        data_l = s - grid_gap
        data_u = s + grid_gap

        # define perturbation
        # perturbation = PerturbationLpNorm(norm=np.inf, x_L=data_l, x_U=data_u)
        # define perturbed data
        # x = BoundedTensor(s, perturbation)

        # lb, ub = model.compute_bounds(x=(x,), method="backward")
        
        # compute lb and ub by sampling
        N_samples = 50
        lb_list = []
        ub_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s) * (data_u - data_l) + data_l
            h_sample = self.h(x_sample)
            lb_list.append(h_sample)
            ub_list.append(h_sample)
        
        lb = torch.stack(lb_list, dim=0).min(dim=0)[0]
        ub = torch.stack(ub_list, dim=0).max(dim=0)[0]
        
        unsatified_index = torch.logical_and(unsafe_mask>0, ub > 0)

        # check if the safe hyperrectangles satisfy cbf condition
        positive_mask =  torch.logical_not(unsafe_mask)

        u_qp, d_min = self.get_control_disturb_vertices(s)

        lb_dx, ub_dx = self.dynamic_system.range_dxdt(data_l, data_u, u_qp)

        # # define perturbation
        # perturbation = PerturbationLpNorm(norm=np.inf, x_L=data_l, x_U=data_u)
        # # define perturbed data
        # x = BoundedTensor(s, perturbation)
        
        # #### A lower upper 
        # required_A = defaultdict(set)
        # required_A[model.output_name[0]].add(model.input_name[0])

        # lb_h, ub_h = model.compute_bounds(x=(x,), method="backward")

        # compute lb and ub by sampling
        N_samples = 50
        lb_list = []
        ub_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s)* (data_u - data_l) + data_l
            h_sample = self.h(x_sample)
            lb_list.append(h_sample)
            ub_list.append(h_sample)
        
        lb_h = torch.stack(lb_list, dim=0).min(dim=0)[0]
        ub_h = torch.stack(ub_list, dim=0).max(dim=0)[0]

        ########### A_J_lower upper

        # jacobian_lower, jacobian_upper = model_jacobian.compute_jacobian_bounds(x)

        # compute jacobian bound by sampling
        N_samples = 50
        jacobian_lower_list = []
        jacobian_upper_list = []
        for i in range(N_samples):
            x_sample = torch.rand_like(s) * (data_u - data_l) + data_l
            hs_sample = self.h(x_sample)
            jacobian_sample = self.jacobian(hs_sample, x_sample)

            jacobian_lower_list.append(jacobian_sample)
            jacobian_upper_list.append(jacobian_sample)

        jacobian_lower = torch.stack(jacobian_lower_list, dim=0).min(dim=0)[0]
        jacobian_upper = torch.stack(jacobian_upper_list, dim=0).max(dim=0)[0]

        # print(f"lower_jacobian is {jacobian_lower}")
        # print(f"upper_jacobian is {jacobian_upper}")
        lb_j = jacobian_lower.reshape(s.shape[0], -1).to(s.device)
        ub_j = jacobian_upper.reshape(s.shape[0], -1).to(s.device)
        # print(f"lb_j is {lb_j}")
        # print(f"ub_j is {ub_j}")

        # compute the multiplication of lower bound and upper bound
        lb_j_lb_dx = torch.mul(lb_j, lb_dx)
        ub_j_ub_dx = torch.mul(ub_j, ub_dx)
        lb_j_ub_dx = torch.mul(lb_j, ub_dx)
        ub_j_lb_dx = torch.mul(ub_j, lb_dx)

        # sovlve convex relaxation problem
        # J, dx, X, h= self._solve_convex_relataxion(lb_j, ub_j, lb_dx, ub_dx, lb_h, ub_h, lb_j_lb_dx, ub_j_ub_dx, lb_j_ub_dx, ub_j_lb_dx)

        # q_min = torch.sum(X, dim=1, keepdim=True) + self.clf_lambda * h

        q_min_1 = torch.sum(lb_j_lb_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_2 = torch.sum(ub_j_ub_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_3 = torch.sum(lb_j_ub_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h
        q_min_4 = torch.sum(ub_j_lb_dx, dim=1, keepdim=True) + self.clf_lambda * lb_h

        q_min = torch.min(torch.min(q_min_1, q_min_2), torch.min(q_min_3, q_min_4))
        
        unsatified_index = torch.logical_or( torch.logical_and(positive_mask>0, q_min < 0.0),  unsatified_index )
       
        return torch.where(unsatified_index.squeeze(dim=-1)>0)[0]

    def cbvf_vi_loss(
        self,
        s: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        hs : torch.Tensor,
        gradh : torch.Tensor,
        grid_gap: torch.Tensor = None,
        model: torch.Tensor = None,
        model_jacobian: torch.Tensor = None,
        coefficients_hji_boundary_loss: float=0.1,
        coefficients_hji_descent_loss: float=2,
        coefficients_hji_inadmissible_loss: float = 2,
        requires_grad: bool = True,
    ) -> List[Tuple[str, torch.Tensor]]:

        loss = []
      
        
        # curricumlum_learning_factor = max(1 - self.current_epoch / (self.trainer.max_epochs -200), -0.1)
        # positive_mask = torch.logical_and((hs >= -0.2), hs >= curricumlum_learning_factor * self.max_value_function.to(s.device)) 
        positive_mask =  torch.logical_not(unsafe_mask).unsqueeze(dim=-1) # (hs >= -0.75) 
         
        #####################################################
        
        baseline = torch.min(self.dynamic_system.state_constraints(s), dim=1, keepdim=True).values.detach() - 0.1
        
        value_fun_violation = (baseline -  hs) * 1
        

        # condition_active = torch.sigmoid(10 * (1.0 + eps - H))

        # u_qp, qp_relaxation = self.solve_CLF_QP(s, gradh,requires_grad=requires_grad, epsilon=0)
        # u_qp = self.solve_CLF_QP(s, requires_grad=requires_grad, epsilon=0)
        # u_qp = torch.ones(s.shape[0], self.dynamic_system.nu, requires_grad=True).float().to(s.device)
        # qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # ####### Minimize the qp relaxation to encourage satisfying the decrease condition #################
        # qp_relaxation_loss = qp_relaxation[torch.logical_not(unsafe_mask)].mean()
        # loss.append(("QP_relaxation", qp_relaxation_loss))

        # Get the current value of the CLBF and its Lie derivatives
        Lf_V, Lg_V, Ld_V= self.V_lie_derivatives(s, gradh)

        u_qp, d_min = self.get_control_disturb_vertices(s)

        if u_qp is not None:
            control_term = torch.bmm(
                            Lg_V[:, :].unsqueeze(1),
                            u_qp.reshape(-1, self.dynamic_system.nu, 1)
                        ) 
        else:
            control_term = 0

        if d_min is not None:
            disturbance_term =  torch.bmm(
                            Ld_V[:, :].unsqueeze(1),
                            d_min.reshape(-1, self.dynamic_system.nd, 1)
                        )
        else:
            disturbance_term = 0
        
        # Use the dynamics to compute the derivative of V
        Vdot =  Lf_V[:, :].unsqueeze(1) + control_term + disturbance_term
                
        Vdot = Vdot.reshape(hs.shape)

        decent_violation_lin = Vdot + self.clf_lambda * hs - 0.5

        # decent_violation_lin = decent_violation_lin[torch.logical_not(unsafe_mask)]

        cbvf_vi_loss, index_min = torch.min(torch.hstack([value_fun_violation, decent_violation_lin]), dim=1)

        cbvf_vi_loss_term = coefficients_hji_descent_loss * torch.abs(cbvf_vi_loss)[positive_mask.squeeze(dim=-1)].mean()
        # hji_vi_loss_term = coefficients_hji_descent_loss * F.relu(-cbvf_vi_loss).mean()
     

        descent_loss_mask =  (index_min > 0)
        boundary_loss_mask = torch.logical_not(descent_loss_mask)
        
        descent_loss_mask = torch.logical_and(descent_loss_mask, positive_mask.squeeze())
        boundary_loss_mask = torch.logical_and(boundary_loss_mask, positive_mask.squeeze())
        
        cbvf_vi_boundary_loss = cbvf_vi_loss[boundary_loss_mask]
        cbvf_vi_descent_loss =  cbvf_vi_loss[descent_loss_mask]
        
        self.hji_vi_boundary_loss_term =  torch.abs(cbvf_vi_boundary_loss).mean().detach()
        self.hji_vi_descent_loss_term = torch.abs(cbvf_vi_descent_loss).mean().detach()
        
        # condition_active = torch.sigmoid(2 * ( - self.hji_vi_boundary_loss_term))
        # decent_violation_lin = decent_violation_lin * condition_active

        # cbvf_vi_loss, index_min = torch.min(torch.hstack([value_fun_violation, decent_violation_lin]), dim=1)
        # hji_vi_loss_term = coefficients_hji_descent_loss * torch.abs(cbvf_vi_loss)[positive_mask.squeeze(dim=-1)].mean()

        # loss.append(("hji_vi_boundary_loss_term", hji_vi_boundary_loss_term))
        # loss.append(("hji_vi_descent_loss_term", hji_vi_descent_loss_term))
        if self.train_mode != 2:
            loss.append(("hji_vi_descent_loss_term", cbvf_vi_loss_term))
            

        #############################################
        # xdot = self.dynamics_model.closed_loop_dynamics(x, u_qp, params=s)

        # x_next = self.dynamic_system.step(s, u_qp)
        # H_next = self.h(x_next)
        # violation = F.relu(
        #     - ((H_next - hs) / self.dynamic_system.dt + self.clf_lambda * hs)
        # )
        # violation = coefficients_hji_descent_loss * violation # * condition_active

        # clbf_descent_term_sim = violation[positive_mask].mean()
        # loss.append(("descent_term_simulated", clbf_descent_term_sim))

        #################################################

        #####################################################

        # unsafe_area_violation = hs[unsafe_mask]
        # unsafe_area_violation_term = coefficients_hji_inadmissible_loss * F.relu(unsafe_area_violation).mean()
        # loss.append(("unsafe_area_hji_term", unsafe_area_violation_term))
        

        ###################################################

        regulation_loss = 50 * F.relu( hs + 0.1 ) # * torch.sigmoid(- 10 * baseline) 
        # regulation_loss_term = regulation_loss[torch.logical_not(unsafe_mask)].mean()
        # regulation_loss = 1 * F.relu( hs -  (hs.detach() - 0.05)) * torch.sigmoid(- 5 * self.hji_vi_boundary_loss_term ) 
        regulation_loss_term = regulation_loss[unsafe_mask].mean()
        loss.append(("regulation_loss_term", regulation_loss_term))

        placeholder_loss = 50 * F.relu( hs - 100000 )
        placeholder_loss_term = placeholder_loss.mean()
        loss.append(("placeholder_loss_term", placeholder_loss_term))


        return loss


    def solve_CLF_QP(
        self,
        x,
        gradh,
        u_ref: Optional[torch.Tensor] = None,
        relaxation_penalty: Optional[float] = 1000,
        requires_grad: bool = False,
        epsilon: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation, defaults to
                                self.clf_relaxation_penalty
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. If provided, must have
                   dimensions bs x self.dynamics_model.n_controls. If not provided,
                   default to calling self.u_reference.
            requires_grad: if True, use a differentiable layer
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # Get the value of the CLF and its Lie derivatives
        H = self.h(x)
        Lf_V, Lg_V, Ld_V = self.V_lie_derivatives(x, gradh)

        # Get the reference control input as well
        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {self.dynamic_system.nu} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == self.dynamic_system.nu, err_message
        else:
            u_ref = self.nominal_controller(x)
            err_message = f"u_ref shouldn't be None!!!!"
            assert u_ref is not None, err_message
        
        if requires_grad:
            # return self._solve_CLF_QP_cvxpylayers(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )

            # return self._solve_CLF_QP_cvxpylayers2(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )

            return self._solve_CLF_QP_OptNet(
                x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            )

            # return self._solve_CLF_QP_OptNet2(
            #     x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
            # )
        else:
            return self._solve_CLF_QP_gurobi(
            x, u_ref, H, Lf_V, Lg_V, relaxation_penalty, epsilon=epsilon
        )
    

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        Gurobi, which does not allow for backpropagation.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # To find the control input, we want to solve a QP constrained by
        #
        # -(L_f V + L_g V u + lambda V) <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        #  -(L_f V + L_g V u + lambda V) - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # We want the objective to be to minimize
        #
        #           ||u||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u + relaxation_penalty * r^2

        n_controls = self.dynamic_system.nu
        
        allow_relaxation = not (relaxation_penalty == float("inf"))

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, 1)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("clf_qp", env=env) as model:
                    
                    # Create variables for control input and (optionally) the relaxations
                    lower_lim, upper_lim = self.dynamic_system.control_limits
                    upper_lim = upper_lim.cpu().numpy()
                    lower_lim = lower_lim.cpu().numpy()
                    u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
                    if allow_relaxation:
                        r = model.addMVar(1, lb=0, ub=GRB.INFINITY)

                    # Define the cost
                    Q = np.eye(n_controls)
                    u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
                    objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
                    if allow_relaxation:
                        relax_penalties = relaxation_penalty * np.ones(1)
                        objective += relax_penalties @ r

                    # Now build the CLF constraints
                
                    Lg_V_np = Lg_V[batch_idx, :].detach().cpu().numpy()
                    Lf_V_np = Lf_V[batch_idx, 0].detach().cpu().numpy()
                    V_np = V[batch_idx, 0].detach().cpu().numpy()
                    clf_constraint = -(Lf_V_np + Lg_V_np @ u + self.clf_lambda * V_np - epsilon)
                    if allow_relaxation:
                        clf_constraint -= r[0]
                    model.addConstr(clf_constraint <= 0.0, name=f"Scenario {0} Decrease")

                    # Optimize!
                    model.setParam("DualReductions", 0)
                    model.setObjective(objective, GRB.MINIMIZE)
                    model.optimize()

                    if model.status != GRB.OPTIMAL:
                        # Make the relaxations nan if the problem was infeasible, as a signal
                        # that something has gone wrong
                        if allow_relaxation:
                            for i in range(1):
                                r_result[batch_idx, i] = torch.tensor(float("nan"))
                        continue

                    # Extract the results
                    for i in range(n_controls):
                        u_result[batch_idx, i] = torch.tensor(u[i].x)
                    if allow_relaxation:
                        for i in range(0):
                            r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)

    def _solve_CLF_QP_OptNet(self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
        epsilon : float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bs = x.shape[0]
        nu = self.dynamic_system.nu
        nv = nu + 1
        nieq = 4

        u_min, u_max = self.dynamic_system.control_limits

        diag_Q =  torch.ones(nv).float().to(x.device)
        diag_Q[-1] = 1000  # 2 * torch.Tensor([1, 1000]).float().to(x.device)
        Q = torch.diag(diag_Q).reshape(nv, nv)
        Q = Variable(Q.expand(bs, nv, nv))
        # print(f"Q shape is {Q.shape} \n")

        p = Variable(torch.zeros(bs, nv).float().to(x.device))
        # print(f"u_ref shape is {u_ref.shape}")
        p[:, 0] = -2 * u_ref.squeeze(-1)
        # print(f"p shape is {p.shape} \n")

        # print(f"Lf_h shape is {Lf_V.shape}")
        # print(f"Lg_h shape is {Lg_V.shape}")
        
        G = Variable(torch.zeros(bs, nieq, nv).to(x.device))
        G[:, 0, 0:nu] = -Lg_V
        G[:, 0, nu:nv] = -1
        G[:, 1, 0:nu] = -1
        G[:, 2, 0:nu] = 1
        G[:, 3, nu:nv] = -1
        # print(f"G shape is {G.shape}")
        # print(f"Lg_V is {Lg_V}")
        # print(f"G is {G}")

        h = Variable(torch.zeros(bs, nieq).to(x.device))
        h[:, 0] = Lf_V.squeeze(-1) + 0.5 * V.squeeze(-1) - epsilon
        h[:, 1] = -u_min
        h[:, 2] = u_max
        h[:, 3] = 0
        # print(f"h shape is {h.shape}")
        # print(f"h is {h}")

        e = Variable(torch.Tensor())

        u_delta = QPFunction(verbose=0)(Q, p, G, h, e, e)
        # print(f"u_delta shape is {u_delta.shape}")
        # print(f"u_delta is {u_delta}")
        
        return u_delta[:, 0:1], u_delta[:, 1:2]
    

    def Dh(self, s, u_vi):
        hs = self.h(s)
        gradh = self.jacobian(hs, s)

        Lf_h, Lg_h, Ld_V = self.V_lie_derivatives(s, gradh)
        u = u_vi.expand(Lg_h.shape[0], -1)
        # u = self.nominal_controller(s)
        Lg_h_u = Lg_h * u
        dt = Lf_h + Lg_h_u.sum(dim=1, keepdim=True)
        return dt

    def gradient_descent_condition(self, s, u_vi, alpha=0.5):
        h = self.h(s)
        dh = self.Dh(s, u_vi)

        result = dh + self.clf_lambda * h
        
        return result


    def get_control_disturb_vertices(self, s):

        assert self.use_h0 == True, "h0 is not defined"
        
        u_qp = None
        d_min = None

        if self.use_h0:
            if self.dynamic_system.nu != 0:
                # find the action that maximize the hamiltonian
                hs_next_list = []
                for u in self.u_v:
                    with torch.no_grad():
                        x_next = self.dynamic_system.step(s, torch.ones(s.shape[0], self.dynamic_system.nu).to(s.device)*u.to(s.device))
                        hs_next = self.h0(x_next)
                        hs_next_list.append(hs_next)

                hs_next = torch.stack(hs_next_list, dim=1)
                _, index_control = torch.max(hs_next, dim=1, keepdim=True)

                index_control = index_control.squeeze()
                u_v = torch.cat(self.u_v, dim=0)
                u_qp = u_v[index_control]
                u_qp = u_qp.reshape(-1, self.dynamic_system.nu).to(s.device)
            

            if self.dynamic_system.nd != 0:
                # find the disturbance that minimize the hamiltonian
                hs_next_list = []
                for d in self.d_v:
                    with torch.no_grad():
                        x_next = self.dynamic_system.step(s, d=torch.ones(s.shape[0], self.dynamic_system.nd).to(s.device)*d.to(s.device))
                        hs_next = self.h0(x_next)
                        hs_next_list.append(hs_next)

                hs_next = torch.stack(hs_next_list, dim=1)
                _, index_control = torch.min(hs_next, dim=1, keepdim=True)

                index_control = index_control.squeeze()
                d_v = torch.cat(self.d_v, dim=0)
                d_min = d_v[index_control]
                d_min = d_min.reshape(-1, self.dynamic_system.nd).to(s.device)
        else:
            raise f"please load guide function first"

        return u_qp, d_min
            

    def on_train_start(self) -> None:
        print(f"############### Training start #########################")

    def on_train_epoch_start(self) -> None:
        print("################## the epoch start #####################")
        self.epoch_start_time = time.time()

        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        s, grid_gap, safe_mask, unsafe_mask, _ = batch
    
        safe_mask = safe_mask.flatten()
        unsafe_mask = unsafe_mask.flatten()

        s.requires_grad_(True)
        
        # s = torch.Tensor([1.23, 0.1]).to(s.device).reshape(-1,2)

        # Compute the losses
        component_losses = {}
       
        s_random = s # + torch.mul(grid_gap , torch.rand(s.shape[0], self.dynamic_system.ns, requires_grad=True).to(s.device)) - grid_gap/2
        
        hs = self.h(s_random)
        gradh = self.jacobian(hs, s_random)

       
        if self.train_mode == 0:
         
            component_losses.update(
                self.cbvf_vi_loss(s_random, safe_mask, unsafe_mask, 
                                hs, gradh, 
                                coefficients_hji_boundary_loss=self.coefficient_for_performance_loss,
                                coefficients_hji_descent_loss=self.coefficients_for_descent_loss,
                                coefficients_hji_inadmissible_loss=self.coefficient_for_inadmissible_state_loss)
            )
            

        if self.train_mode == 1:
           
            component_losses.update(
                self.cbvf_vi_loss(s_random, safe_mask, unsafe_mask, 
                                hs, gradh, 
                                coefficients_hji_boundary_loss=self.coefficient_for_performance_loss,
                                coefficients_hji_descent_loss=self.coefficients_for_descent_loss,
                                coefficients_hji_inadmissible_loss=self.coefficient_for_inadmissible_state_loss)
            )



        # if self.train_mode == 2:
            
        #     component_losses.update(
        #         self.cbvf_vi_loss(s_random, safe_mask, unsafe_mask, 
        #                         hs, gradh, 
        #                         coefficients_hji_boundary_loss=self.coefficient_for_performance_loss,
        #                         coefficients_hji_descent_loss=self.coefficients_for_descent_loss,
        #                         coefficients_hji_inadmissible_loss=self.coefficient_for_inadmissible_state_loss)
        #     )

        #     if (self.current_epoch+1) % self.trainer.reload_dataloaders_every_n_epochs == 0:
        #         # copy_h_bound = copy.deepcopy(self.h)
        #         # model = BoundedModule(copy_h_bound, s)
        #         # model.train()

        #         # copy_h_j =  copy.deepcopy(self.h)
        #         # model_jacobian = BoundedModule(copy_h_j, s,
        #         #                                 bound_opts={
        #         #                                                 'sparse_features_alpha': False,
        #         #                                                 'sparse_spec_alpha': False,
        #         #                                             })
        #         # model_jacobian.augment_gradient_graph(s)
        #         # model_jacobian.train()
        #         with torch.no_grad():
        #             model = None
        #             model_jacobian = None

        #             component_losses.update(
        #                 self.epsilon_decent_loss(
        #                     s, safe_mask, unsafe_mask, grid_gap, hs, gradh, model, model_jacobian, coefficients_descent_loss=1
        #                 )
        #             )
        #             component_losses.update(
        #                 self.epsilon_area_loss(s, safe_mask, unsafe_mask, grid_gap,model, coefficients_unsafe_epsilon_loss=100)
        #             )
        #         # del copy_h_bound
        #         # del copy_h_j

        if self.train_mode == 3:
            
            component_losses.update(
                self.cbvf_vi_loss(s_random, safe_mask, unsafe_mask, 
                                hs, gradh, 
                                coefficients_hji_boundary_loss=self.coefficient_for_performance_loss,
                                coefficients_hji_descent_loss=self.coefficients_for_descent_loss,
                                coefficients_hji_inadmissible_loss=self.coefficient_for_inadmissible_state_loss)
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(s)
        safety_loss = torch.tensor(0.0).type_as(s)
        performance_loss = torch.tensor(0.0).type_as(s)
        descent_loss = torch.tensor(0.0).type_as(s)

        # For the objectives, we can just sum them
        for key, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value
            if key in self.performance_lose_name:
                performance_loss += loss_value
            if key in self.safety_loss_name:
                safety_loss += loss_value
            if key in self.descent_loss_name:
                descent_loss += loss_value
        

        batch_dict = {"loss": total_loss, **component_losses}

        
        return batch_dict
    

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list

        self.epoch_end_time = time.time()

        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        descent_losses = 0.0
        safety_losses = 0.0
        performance_losses = 0.0
        
        
        for key in avg_losses.keys():
            if key in self.safety_loss_name:
                safety_losses += avg_losses[key]
            if key in self.performance_lose_name:
                performance_losses += avg_losses[key]
            if key in self.descent_loss_name:
                descent_losses += avg_losses[key]
        
        average_loss = safety_losses + performance_losses + descent_losses


        # Log the overall loss...
        # if self.current_epoch > self.learn_shape_epochs:
        self.log("Total_loss/train", average_loss)
        print_info(f"\n the overall loss of this training epoch {average_loss}\n")
        self.log("Epoch_time/train", self.epoch_end_time - self.epoch_start_time)
        print_info(f"the epoch time consume is {self.epoch_end_time - self.epoch_start_time}")
        # self.log("Safety_loss/train", safety_losses)
        # print(f"overall safety loss of this training epoch {safety_losses}")
        # self.log("Descent_loss/train", descent_losses)
        # print(f"overall descent loss of this training epoch {descent_losses}")
        # self.log("Performance_loss/train", performance_losses)
        # print(f"overall performance loss of this training epoch {performance_losses}")
        # print(f"the adaptive coefficient is {self.coefficients_for_descent_loss}")

        # print(f"hji_vi boundary loss is  {self.hji_vi_boundary_loss_term}")
        # print(f"hji_vi descent loss is  {self.hji_vi_descent_loss_term}")

        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + "/train", avg_losses[loss_key], sync_dist=True)

           
        if self.train_mode == 0 and (performance_losses < 1 and safety_losses < 1): # warm up
            self.trainer.should_stop = True
            # pass
        

        if self.train_mode == 1 and (safety_losses < 1e-3 and descent_losses < 1e-2 ):
            self.trainer.should_stop = True

        if self.train_mode == 2 and ( safety_losses < 1e-2 and descent_losses < 1e-2 ) :
            # self.trainer.should_stop = True
            pass
        
        if self.train_mode == 1 and (self.current_epoch + 1) % (self.trainer.reload_dataloaders_every_n_epochs ) == 0:
            self.data_module.augment_dataset()
            torch.save(self.data_module.s_training, "s_training.pt")
            

        if (self.train_mode == 2 or self.train_mode == 3 ) and (self.current_epoch + 1) % (self.trainer.reload_dataloaders_every_n_epochs ) == 0:
            self.data_module.augment_dataset()
            # torch.save(self.data_module.s_training, "s_training.pt")
            if self.data_module.verified == 1:
                print("verified!!!!!!!!!!!")
                self.trainer.should_stop = True
            elif self.data_module.verified == 0:
                print("reach minimum gap!!!!!!!!!!!")
                self.trainer.should_stop = True
            else:
                pass               
        
        print_info(f"current learning rate is {self.trainer.optimizers[0].param_groups[0]['lr']}")

    def on_train_end(self) -> None:

        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
    
        x, _, safe_mask, unsafe_mask, _ = batch
        safe_mask = safe_mask.flatten()
        unsafe_mask = unsafe_mask.flatten()
        
        # Get the various losses
        batch_dict = {"shape_h": {}, "unsafe_violation": {}, "descent_violation": {}, "inadmissible_boundary":{}, "inadmissible_area": {}, "admissible_area": {} }
        x.requires_grad_(True)
        
        # record shape_h
        h_x = self.h(x)
        gradh = self.jacobian(h_x, x)
        
        
        batch_dict["shape_h"]["state"] = x
        batch_dict["shape_h"]["val"] = h_x

        # record unsafe_violation
        h_x_neg_mask = h_x < -2
        h_x_pos_mask = h_x > 0
        unit_index = torch.hstack((unsafe_mask.unsqueeze(dim=1), h_x_pos_mask))

        h_x_unsafe_violation_indx = torch.all(unit_index, dim=1)
        ##  record states and its value
        s_unsafe_violation = x[h_x_unsafe_violation_indx]
        h_s_unsafe_violation = h_x[h_x_unsafe_violation_indx]

        batch_dict["unsafe_violation"]["state"] = s_unsafe_violation
        batch_dict["unsafe_violation"]["val"] = h_s_unsafe_violation

        # record descent_violation
        c_list = [ self.gradient_descent_condition(x, u_i.to(x.device)) for u_i in self.u_v ]
        c_list = torch.hstack(c_list)

        descent_violation, u_star_index = torch.max(c_list, dim=1)

        descent_violation_mask = torch.logical_and(descent_violation < 0, torch.logical_not(unsafe_mask.squeeze(dim=-1)))  

        s_descent_violation = x[descent_violation_mask]
        u_star_index_descent_violation = u_star_index[descent_violation_mask]
        h_s_descent_violation = h_x[descent_violation_mask]
        Lf_h_descent_violation, Lg_h_descent_violation, _ = self.V_lie_derivatives(s_descent_violation, gradh[descent_violation_mask])

        batch_dict["descent_violation"]["state"] = s_descent_violation
        batch_dict["descent_violation"]["val"] = h_s_descent_violation
        batch_dict["descent_violation"]["Lf_h"] =  Lf_h_descent_violation
        batch_dict["descent_violation"]["Lg_h"] = Lg_h_descent_violation
        batch_dict["descent_violation"]["u_star_index"] = u_star_index_descent_violation

        # get safety boundary
        baseline = torch.min(self.dynamic_system.state_constraints(x), dim=1, keepdim=True).values
        
        inadmissible_boundary_index = torch.logical_and(baseline < 0, baseline > -0.05).squeeze(dim=-1)
        inadmissible_boundary_state = x[inadmissible_boundary_index]

        batch_dict["inadmissible_boundary"]["state"] = inadmissible_boundary_state

        inadmissible_area_index = unsafe_mask
        inadmissible_area_state = x[inadmissible_area_index]

        batch_dict["inadmissible_area"]["state"] = inadmissible_area_state

        admissible_area_index = torch.logical_not(unsafe_mask)
        admissible_area_state = x[admissible_area_index]

        batch_dict["admissible_area"]["state"] = admissible_area_state



        return batch_dict

    def test_epoch_end(self, outputs):
        
        print("test_epoch_end")
        save_dir = self.trainer.default_root_dir + "/test_results.pt" 
        torch.save(outputs, save_dir)
        
        print(f"results is save to {save_dir}")
        


    def configure_optimizers(self):
        clbf_params = list(self.h.parameters()) # list(self.g.parameters())
        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.learning_rate,
        )
        lr_scheduler = ExponentialLR(clbf_opt, gamma=0.995)
        self.opt_idx_dict = {0: "clbf"}

        return {"optimizer":clbf_opt, "lr_scheduler": lr_scheduler}



if __name__ == "__main__":

    
    current_date_str = datetime.datetime.now().strftime("%d_%b")

    train_mode = 0
    system = inverted_pendulum_1
    default_root_dir = "logs/CBF_logs/IP_" + current_date_str
    checkpoint_dir = "saved_models/inverted_pendulum_stage_1/checkpoints/epoch=293-step=2646.ckpt"

    grid_gap = torch.Tensor([0.2, 0.2])  

    ########################## start training ###############################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(5e5), train_mode=train_mode)

    NN = NeuralCBF(dynamic_system=system, data_module=data_module, train_mode=train_mode)
    NN0 = NeuralCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module)
    NN.set_previous_cbf(NN0.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=50,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)



