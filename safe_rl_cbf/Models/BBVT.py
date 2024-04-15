from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.Verifier import Verifier
from safe_rl_cbf.Models.Learner import Learner
from safe_rl_cbf.Dynamics.control_affine_system import ControlAffineSystem
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Analysis.draw_cbf import draw_cbf
from safe_rl_cbf.Analysis.Plotter import Plotter

class BBVT:
    def __init__(self, 
                model: pl.LightningModule,
                prefix: str = "",
                log_dir: str = "logs",
                training_points_num = 1e5,
                train_batch_size = 64,
                testing_points_num = 1e5,
                test_batch_size = 64,
                test_index = {"0": "None", "1": "None"},
                initial_grid_gap = [0.5, 0.5],
                minimum_grip_gap = 0.005,
                verify_batch_size = 64,
                visualize_index = [0, 1],
                ):
        self.model = model
        date = datetime.datetime.now().strftime("%d_%b")
        self.prefix = prefix
        self.log_dir = os.path.join(log_dir, self.prefix) 
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.learner = Learner(model=model, prefix=prefix, log_dir=self.log_dir, 
                               training_points_num=training_points_num, train_batch_size=train_batch_size, 
                               testing_points_num=testing_points_num, test_batch_size=test_batch_size, test_index=test_index)

        self.verifier = Verifier(model=model, initial_grid_gap=initial_grid_gap, minimum_grip_gap=minimum_grip_gap,
                                  verify_batch_size=verify_batch_size, prefix=prefix, log_dir=self.log_dir)

        self.plotter = Plotter(system=self.model.dynamic_system, prefix=prefix, log_dir=self.log_dir, x_index=visualize_index[0], y_index=visualize_index[1])

        self.data_prepared = False

    def prepare_data(self, training_points_num = None, train_batch_size = None,
                        testing_points_num = None , test_batch_size=None , test_index = None,
                        initial_grid_gap = None, verify_batch_size = None):
        assert training_points_num is not None or self.learner.training_points_num is not None, "Please specify the number of training points"
        training_points_num = training_points_num if training_points_num is not None else self.learner.training_points_num

        assert train_batch_size is not None or self.learner.train_batch_size is not None, "Please specify the batch size"
        train_batch_size = train_batch_size if train_batch_size is not None else self.learner.train_batch_size

        assert testing_points_num is not None or self.learner.testing_points_num is not None, "Please specify the number of testing points"
        testing_points_num = testing_points_num if testing_points_num is not None else self.learner.testing_points_num

        assert test_batch_size is not None or self.learner.test_batch_size is not None, "Please specify the batch size"
        test_batch_size = test_batch_size if test_batch_size is not None else self.learner.test_batch_size

        assert test_index is not None or self.learner.test_index is not None, "Please specify the test index"
        test_index = test_index if test_index is not None else self.learner.test_index

        assert initial_grid_gap is not None or self.verifier.initial_grid_gap is not None, "Please specify the initial grid gap"
        initial_grid_gap = initial_grid_gap if initial_grid_gap is not None else self.verifier.initial_grid_gap
        assert verify_batch_size is not None or self.verifier.verify_batch_size is not None, "Please specify the batch size"

        self.learner.prepare_data(training_points_num, train_batch_size, testing_points_num, test_batch_size, test_index)
        self.verifier.prepare_data(initial_grid_gap, verify_batch_size)
        
        self.data_prepared = True

    def training_and_verifying(self, max_epochs=20, training_without_verification_epochs=10, k=5):
        assert self.data_prepared, "Please prepare for training and verification first"

        save_ce = False
        epochs_count = 0

        self.learner.pretrain(epochs=training_without_verification_epochs)
        epochs_count = epochs_count + training_without_verification_epochs

        while max_epochs > epochs_count:    

            verified_flag = self.verifier.verify()
            
            if verified_flag:
                    print_info(f"verified =  {verified_flag}")
                    break
            else:
                print_info(f" augment training data nums = {len(self.verifier.verify_data_module)} ")
                self.learner.augment_dataset(self.verifier.verify_data_module)
                print_info(f"after augment training data nums = {len(self.learner.train_data_module)} ")
                self.verifier.prepare_data()
                if len(self.verifier.augment_data_module) > 1e8:
                    print_info(f"reset the training and verification data modules. epochs = {epochs_count}")
                    self.learner.reset()
                    self.verifier.reset()

                print_warning(f" verified =  {verified_flag}")
                print_warning(f" epochs =  {epochs_count}")

            if not save_ce:
                self.verifier.augment_data_module.save_as_tensor(file_name="s_training.pt", num=1e5)
                save_ce = True
                
            self.learner.train(epochs=k)
            epochs_count = epochs_count + k

    def test(self):
        self.learner.test()

    def draw_figures(self):

        # check if test result is available
        self.plotter.prepare_for_drawing()
    
        self.plotter.draw_cbf_2d()
        self.plotter.draw_decent_violation()
        self.plotter.draw_counterexamples()


    def save(self):
        pass
