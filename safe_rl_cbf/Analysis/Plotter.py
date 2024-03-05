from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem

class Plotter:
    def __init__(self,
                 system: ControlAffineSystem,
                 prefix: str = "",
                 log_dir: str = "logs",
                 x_index = 0,
                 y_index = 1,
                 ):
        self.system = system
        self.prefix = prefix
        self.log_dir = log_dir

        self.x_index = x_index
        self.y_index = y_index

        self.domain_limit_lb, self.domain_limit_ub = self.system.domain_limits

    def prepare_for_drawing(self):
        if not os.path.exists( os.path.join(self.log_dir, "fig")):
            os.makedirs(os.path.join(self.log_dir, "fig") )

        self.fig_dir = os.path.join(self.log_dir, "fig")

        assert os.path.exists( self.log_dir + "/test_results.pt"), "Please run the test first"
        test_results = torch.load( self.log_dir + "/test_results.pt")

        h_shape_s = []
        h_shape_val = []
        s_unsafe_violation = []
        s_unsafe_violation_val = []

        descent_violation = []

        inadmissible_boundary_state = []

        for batch_id in range(len(test_results)):
            h_shape_s.append(test_results[batch_id]["shape_h"]["state"])
            h_shape_val.append(test_results[batch_id]["shape_h"]["val"])
            s_unsafe_violation.append(test_results[batch_id]["unsafe_violation"]["state"])
            descent_violation.append(test_results[batch_id]["descent_violation"]["state"])
            inadmissible_boundary_state.append(test_results[batch_id]["inadmissible_boundary"]["state"])


        self.h_shape_s = torch.vstack(h_shape_s)
        self.h_shape_val = torch.vstack(h_shape_val)
        self.s_unsafe_violation = torch.vstack(s_unsafe_violation)
        self.descent_violation = torch.vstack(descent_violation)
        self.inadmissible_boundary_state = torch.vstack(inadmissible_boundary_state)
        
        assert os.path.exists("RA_result/extraOuts.mat"), "Please have RA_result/extraOuts.mat in the directory"
        mat_contents = sio.loadmat("RA_result/extraOuts.mat")

        self.hVS_XData = mat_contents['a0']
        self.hVS_YData = mat_contents['a1']
        self.hVS_ZData = mat_contents['a2']
        self.hVS0_XData = mat_contents['a3']
        self.hVS0_YData = mat_contents['a4']
        self.hVS0_ZData = mat_contents['a5']



    def draw_cbf_2d(self, x_index=None, y_index=None):

    
        x_index = x_index if x_index is not None else self.x_index
        y_index = y_index if y_index is not None else self.y_index

        assert self.h_shape_s is not None, "Please prepare for drawing first"
        assert self.h_shape_val is not None, "Please prepare for drawing first"
        assert self.inadmissible_boundary_state is not None, "Please prepare for drawing first"

        X = self.h_shape_s[:, x_index].detach().cpu().numpy()
        Y = self.h_shape_s[:, y_index].detach().cpu().numpy()
        H = self.h_shape_val.squeeze(dim=1).detach().cpu().numpy()

        H_positive_mask = H > 0


        x = X[H_positive_mask]
        y = Y[H_positive_mask]

        plt.figure()

        # Create contour lines or level curves using matpltlib.pyplt module
        # contours = plt.contourf(hVS_XData, hVS_YData, hVS_ZData, levels=[-0.1, 0, 1], colors=['w','y','w'], extend='both')

        # contours2 = plt.contour(hVS0_XData, hVS0_YData, hVS0_ZData, levels=[0], colors='grey', linewidth=5)


        plt.scatter(x, y, s=1, c='b')

        X_in = self.inadmissible_boundary_state[:, x_index].detach().cpu().numpy()
        Y_in = self.inadmissible_boundary_state[:, y_index].detach().cpu().numpy()
        plt.scatter(X_in, Y_in, s=1, c='y')

        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.xlim(self.domain_limit_lb[x_index], self.domain_limit_ub[x_index])
        plt.ylim(self.domain_limit_lb[y_index], self.domain_limit_ub[y_index])
        plt.title("shape of 0-superlevel set")


        legend_elements = [
                            Line2D([0], [0], color='grey', lw=2, label='Obstacles'),
                            Patch(facecolor='y', edgecolor='y',
                                label='Invariant Set from RA'),
                            Patch(facecolor='b', edgecolor='b',
                                label='Invariant Set from neural CBF')
                        ]

        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1.1),loc='upper right')

        assert self.fig_dir is not None, "Please prepare for drawing first"

        plt.savefig( os.path.join(self.fig_dir, "shape_of_cbf.png"))

    def draw_decent_violation(self, x_index=None, y_index=None):

        x_index = x_index if x_index is not None else self.x_index
        y_index = y_index if y_index is not None else self.y_index
        
        
        assert self.descent_violation is not None, "Please prepare for drawing first"
        assert self.s_unsafe_violation is not None, "Please prepare for drawing first"
        assert self.inadmissible_boundary_state is not None, "Please prepare for drawing first"
        assert self.h_shape_s is not None, "Please prepare for drawing first"
        assert self.h_shape_val is not None, "Please prepare for drawing first"

        X_unsafe_violation = self.s_unsafe_violation[:, x_index].detach().cpu().numpy()
        Y_unsafe_violation = self.s_unsafe_violation[:, y_index].detach().cpu().numpy()

        X_descent = self.descent_violation[:, x_index].detach().cpu().numpy()
        Y_descent = self.descent_violation[:, y_index].detach().cpu().numpy()

        print_warning(f"descent violation: {Y_descent.shape[0]}")

        plt.figure()

        X = self.h_shape_s[:, x_index].detach().cpu().numpy()
        Y = self.h_shape_s[:, y_index].detach().cpu().numpy()
        H = self.h_shape_val.squeeze(dim=1).detach().cpu().numpy()

        H_positive_mask = H > 0


        x = X[H_positive_mask]
        y = Y[H_positive_mask]

        X_in = self.inadmissible_boundary_state[:, x_index].detach().cpu().numpy()
        Y_in = self.inadmissible_boundary_state[:, y_index].detach().cpu().numpy()

        plt.scatter(x, y, s=1, c='b')
        plt.scatter(X_in, Y_in, s=1, c='y')
        plt.scatter(X_descent, Y_descent, s=1, c='r')
        plt.scatter(X_unsafe_violation, Y_unsafe_violation, s=1, c='r')
        


        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.xlim(self.domain_limit_lb[x_index], self.domain_limit_ub[x_index])
        plt.ylim(self.domain_limit_lb[y_index], self.domain_limit_ub[y_index])
        plt.title("shape of 0-superlevel set")

        assert self.fig_dir is not None, "Please prepare for drawing first"
        plt.savefig( os.path.join(self.fig_dir, "descent_violation.png"))


    def draw_counterexamples(self, x_index=None, y_index=None):

        x_index = x_index if x_index is not None else self.x_index
        y_index = y_index if y_index is not None else self.y_index

        assert os.path.exists( os.path.join(self.log_dir,"s_training.pt")), "Please store the counter example first"
        s_training = torch.load( os.path.join(self.log_dir,"s_training.pt") )

        X = self.h_shape_s[:, x_index].detach().cpu().numpy()
        Y = self.h_shape_s[:, y_index].detach().cpu().numpy()
        H = self.h_shape_val.squeeze(dim=1).detach().cpu().numpy()

        H_positive_mask = H > 0

        x = X[H_positive_mask]
        y = Y[H_positive_mask]

        X_in = self.inadmissible_boundary_state[:, x_index].detach().cpu().numpy()
        Y_in = self.inadmissible_boundary_state[:, y_index].detach().cpu().numpy()

        X_descent = self.descent_violation[:, x_index].detach().cpu().numpy()
        Y_descent = self.descent_violation[:, y_index].detach().cpu().numpy()

        plt.figure()
        plt.scatter(x, y, s=1, c='b')
        plt.scatter(X_descent, Y_descent, s=1, c='r')
        plt.scatter(X_in, Y_in, s=1, c='y')


        plt.scatter(s_training[:,0], s_training[:,1], marker='X', s=10, c='k')


        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.xlim(self.domain_limit_lb[x_index], self.domain_limit_ub[x_index])
        plt.ylim(self.domain_limit_lb[y_index], self.domain_limit_ub[y_index])
        plt.title("shape of 0-superlevel set")

        plt.savefig( os.path.join(self.fig_dir,  "shape_of_cbf_with_counterexamples.png"))

    def draw_cbf_contour(self, x_index=None, y_index=None):

        x_index = x_index if x_index is not None else self.x_index
        y_index = y_index if y_index is not None else self.y_index

        fig1,ax1=plt.subplots(1,1)

        X = self.h_shape_s[:, x_index].detach().cpu().numpy()
        Y = self.h_shape_s[:, y_index].detach().cpu().numpy()
        H = self.h_shape_val.squeeze(dim=1).detach().cpu().numpy()

        cp = ax1.contourf(X.reshape((math.gcd(X.shape[0], 1000), -1)), Y.reshape((math.gcd(X.shape[0], 1000), -1)), H.reshape((math.gcd(X.shape[0], 1000), -1)))
        fig1.colorbar(cp) # Add a colorbar to a plot
        ax1.set_title('Filled Contours Plot')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$\dot{\theta}$")
        ax1.set_title("contour of CBF")
        
        assert self.fig_dir is not None, "Please prepare for drawing first"
        plt.savefig( os.path.join(self.fig_dir, "contour_of_cbf.png"))
