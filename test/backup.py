# _, J_s = self.V_with_jacobian(s)
        # # print(f"jacobian at s is {J_s}")
        # # print(f"jacobian at s shape is {J_s.shape}")

        # ########## A_f_lower
        # ns = self.dynamic_system.ns
        # A_f_lower = torch.zeros(bs, ns, ns).to(s.device)
        # A_f_lower[:, 0, 1] = 1
        # A_f_lower[:, 1, 0] = self.dynamic_system.gravity * torch.cos(s[:, 0]) / self.dynamic_system.L
        # A_f_lower[:, 1, 1] = -self.dynamic_system.b /(self.dynamic_system.m* self.dynamic_system.L ** 2)
        # # print(f"A_f_lower is {A_f_lower}")
        # # print(f"A_f_lower shape is {A_f_lower.shape}")

        # b_f_lower = torch.zeros(bs, ns, 1).to(s.device)
        # b_f_lower[:, 0, 0] = 0
        # b_f_lower[:, 1, 0] = self.dynamic_system.gravity*( torch.sin(s[:, 0]) - torch.cos(s[:, 0])*s[:, 0] -gridding_gap[:, 0]) /self.dynamic_system.L
        # # print(f"b_f_lower is {b_f_lower}")

        # s_eps = s - gridding_gap
        # # s_eps_p = s + gridding_gap
        # # f_s_0 = self.dynamic_system.f(s_eps)
        # # print(f"f_s_0 is {f_s_0}")
        # f_s_lower = torch.bmm(A_f_lower, s_eps.unsqueeze(dim=-1)) + b_f_lower
        # print(f"f_s_lower = {f_s_lower.squeeze(dim=-1)}")
        # # f_s_lower_p = torch.bmm(A_f_lower, s_eps_p.unsqueeze(dim=-1)) + b_f_lower
        # # print(f"f_s_lower_p = {f_s_lower_p.squeeze(dim=-1)}")

        # ###### A_f_upper
        # A_f_upper = torch.zeros(bs, ns, ns).to(s.device)
        # A_f_upper[:, 0, 1] = 1
        # A_f_upper[:, 1, 0] = self.dynamic_system.gravity * torch.cos(s[:, 0]) / self.dynamic_system.L
        # A_f_upper[:, 1, 1] = -self.dynamic_system.b /(self.dynamic_system.m* self.dynamic_system.L ** 2)
        # # print(f"A_f_upper is {A_f_upper}")
        # # print(f"A_f_upper shape is {A_f_upper.shape}")

        # b_f_upper = torch.zeros(bs, ns, 1).to(s.device)
        # b_f_upper[:, 0, 0] = 0
        # b_f_upper[:, 1, 0] = self.dynamic_system.gravity*( torch.sin(s[:, 0]) - torch.cos(s[:, 0])*s[:, 0] + gridding_gap[:, 0]) /self.dynamic_system.L
        # # print(f"b_f_upper is {b_f_upper}")

        # # f_s_upper = torch.bmm(A_f_upper, s_eps.unsqueeze(dim=-1)) + b_f_upper
        # # print(f"f_s_upper = {f_s_upper.squeeze(dim=-1)}")

        # ####### A_g_lower upper
        # # g_s_0 = self.dynamic_system.g(s)
        # # print(f"g_s_0 = {g_s_0}")
        # # print(f"g_s_0 shape = {g_s_0.shape}")
        # # print(f"u_qp shape is {u_qp.shape}")
        # # print(f"u_qp = {u_qp}")
        # A_g_lower = torch.zeros(bs, ns, ns).to(s.device)
        # A_g_upper = torch.zeros(bs, ns, ns).to(s.device)
        # b_g_lower = torch.bmm(torch.tensor([0, 1/(self.dynamic_system.m * self.dynamic_system.L**2)]).reshape(ns, 1).expand(bs, ns, 1).to(s.device), u_qp.detach().unsqueeze(dim=-1))
        # b_g_upper = torch.bmm(torch.tensor([0, 1/(self.dynamic_system.m * self.dynamic_system.L**2)]).reshape(ns, 1).expand(bs, ns, 1).to(s.device), u_qp.detach().unsqueeze(dim=-1))
        
        # # print(f"b_g_lower is{b_g_lower}")
        # # print(f"b_g_lower shape is{b_g_lower.shape}")

        # ###### start optimization 
        # nv = 9
        # nieq = 18
        # diag_Q = 0.001 * torch.ones(nv).float().to(s.device)
        # Q = torch.diag(diag_Q)
        # # Q[2:4, 6:8] = torch.eye(ns) * 0.5
        # # Q[4:6, 6:8] = torch.eye(ns) * 0.5
        # # Q[6:8, 2:4] = torch.eye(ns) * 0.5
        # # Q[6:8, 4:6] = torch.eye(ns) * 0.5
        # # print(f"Q = {Q}")

        # Q = Variable(Q.expand(bs, nv, nv))
        # #print(f"Q shape is {Q.shape}")

        # p = torch.zeros(bs, nv).to(s.device)
        
        # p[:, 2:4] = J_s.detach().squeeze(dim=1)
        # p[:, 4:6] = J_s.detach().squeeze(dim=1)
        # p[:, 8] = 0.5
        # # print(f"p shape is {p.shape}")
        # # print(f"p = {p}")

        # ns = self.dynamic_system.ns
        # G = Variable(torch.zeros(bs, nieq, nv).to(s.device))
        # # G[:, 0:2, 6:8] = -torch.eye(ns).to(s.device)
        # # G[:, 2:4, 6:8] = torch.eye(ns).to(s.device)
        # G[:, 4:6, 0:2] = A_f_lower 
        # G[:, 4:6, 2:4] = -torch.eye(ns).to(s.device)
        # G[:, 6:8, 0:2] = -A_f_upper
        # G[:, 6:8, 2:4] = torch.eye(ns).to(s.device)
        # G[:, 8:10, 0:2] = A_g_lower
        # G[:, 8:10, 4:6] = -torch.eye(ns).to(s.device)
        # G[:, 10:12, 0:2] = -A_g_upper
        # G[:, 10:12, 4:6] = torch.eye(ns).to(s.device)
        # G[:, 12:13, 0:2] = A_lower
        # G[:, 12:13, 8:9] = -1
        # G[:, 13:14, 0:2] = -A_upper
        # G[:, 13:14, 8:9] = 1
        # G[:, 14:16, 0:2] = -torch.eye(ns).to(s.device)
        # G[:, 16:18, 0:2] = torch.eye(ns).to(s.device)
        # # print(f"G is {G}")
        # # print(f"G shape is {G.shape}")

        # h = Variable(torch.zeros(bs, nieq).to(s.device))
        # # h[:, 0:2] = -lb_j.squeeze(dim=-1)
        # # h[:, 2:4] = ub_j.squeeze(dim=-1)
        # h[:, 4:6] = -b_f_lower.squeeze(dim=-1)
        # h[:, 6:8] = b_f_upper.squeeze(dim=-1)
        # h[:, 8:10] = -b_g_lower.squeeze(dim=-1)
        # h[:, 10:12] = b_g_upper.squeeze(dim=-1)
        # h[:, 12] = -b_lower.squeeze(dim=-1)
        # h[:, 13] = b_upper.squeeze(dim=-1)
        # h[:, 14:16] = -s + gridding_gap
        # h[:, 16:18] = s + gridding_gap
        # # print(f"h is {h}")
        # # print(f"h shape is {h.shape}")
        # e = Variable(torch.Tensor().to(s.device))

        # # convex relaxation solver
        # params = []
        # params.append(data_l)
        # params.append(data_u)
        # params.append(lb_j)
        # params.append(ub_j)
        # params.append(A_f_lower)
        # params.append(b_f_lower.squeeze(dim=-1))
        # params.append(A_f_upper)
        # params.append(b_f_upper.squeeze(dim=-1))

        # result = self.convex_relaxation_solver(
        #     *params,
        #     solver_args={"max_iters": 50000000},
        # )

        # print(f"f = {result[2]}")
        # print(f"X_jf = {result[3]}")

        

        # x_min = QPFunction()(Q, p, G, h, e, e)
        # if  torch.all(torch.isnan(x_min)==False):
        #     s_min = x_min[:,0:2]
        #     f_min = x_min[:, 2:4]
        #     gu_min = x_min[:, 4:6]
        #     J_min = x_min[:, 6:8]
        #     h_min = x_min[:, 8]
        #     # print(f"s = {s}")
        #     # print(f"s_min = {s_min} \n f_min = {f_min} \n gu_min = {gu_min} \n J_min = {J_min}  \n h_min = {h_min}")
            
        #     q_min = torch.bmm(J_s, f_min.unsqueeze(dim=-1) + gu_min.unsqueeze(dim=-1)).squeeze(dim=-1) + 0.5*h_min.unsqueeze(dim=1)
        #     # print(f"q is {Vdot + self.clf_lambda * H}")
        #     # print(f"min of q is {q_min}")

        #     # q_min = q_min.detach()
        #     violation = coefficients_descent_loss * F.relu(-q_min)
        #     # violation = F.relu(h_min)
        #     epsilon_area_q_min_loss_term = violation[torch.logical_not(unsafe_mask)].mean()
            
        #     loss.append(("descent_term_epsilon_area", epsilon_area_q_min_loss_term))
        # else:
        #     print(f"the OptNet has no solution!!!!!!!!!!!")
        #     exit()

        # print(f"qp_relaxation_loss: {qp_relaxation_loss}")
        # print(f"descent_term_lin: {clbf_descent_term_lin}")
        # print(f"descent_term_epsilon_area: {epsilon_area_q_min_loss_term}")