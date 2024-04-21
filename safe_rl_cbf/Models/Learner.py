from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Dataset.TrainingDataModule import TrainingDataModule
from safe_rl_cbf.Dataset.TestingDataModule import TestingDataModule
from safe_rl_cbf.Dataset.VerificationDataModule import VerificationDataModule

class Learner:
    def __init__(self,
                model: pl.LightningModule,
                prefix: str = "",
                log_dir: str = "logs",
                training_points_num = 1e5,
                train_batch_size = 64,
                testing_points_num = 1e5,
                test_batch_size = 64,
                test_index = {"0": "None", "1": "None"},
                 ):
        
        self.model = model
        self.prefix = prefix
        self.log_dir = log_dir
        self.training_points_num = training_points_num
        self.train_batch_size = train_batch_size
        self.testing_points_num = testing_points_num
        self.test_batch_size = test_batch_size
        self.test_index = test_index

       
        self.train_data_module = TrainingDataModule(system=self.model.dynamic_system, train_batch_size=train_batch_size, training_points_num=training_points_num,
                                                     prefix=self.prefix + "training", log_dir=self.log_dir)   
        self.test_data_module = TestingDataModule(system=self.model.dynamic_system, testing_points_num=testing_points_num, test_index= test_index , test_batch_size=test_batch_size, prefix=self.prefix + "testing", log_dir=self.log_dir)    
    
    def reset(self):
        self.train_data_module.reset()
        self.test_data_module.reset()
        

    def prepare_data(self, training_points_num = None, train_batch_size = None,
                      testing_points_num = None , test_batch_size=None , test_index = None):
        
        assert training_points_num is not None or self.training_points_num is not None, "Please specify the number of training points"
        training_points_num = training_points_num if training_points_num is not None else self.training_points_num

        assert train_batch_size is not None or self.train_batch_size is not None, "Please specify the batch size"
        train_batch_size = train_batch_size if train_batch_size is not None else self.train_batch_size

        assert testing_points_num is not None or self.testing_points_num is not None, "Please specify the number of testing points"
        testing_points_num = testing_points_num if testing_points_num is not None else self.testing_points_num

        assert test_batch_size is not None or self.test_batch_size is not None, "Please specify the batch size"
        test_batch_size = test_batch_size if test_batch_size is not None else self.test_batch_size

        assert test_index is not None or self.test_index is not None, "Please specify the test index"
        test_index = test_index if test_index is not None else self.test_index

        self.prepare_for_training(training_points_num, train_batch_size)
        self.prepare_for_testing(testing_points_num, test_batch_size, test_index)

    def prepare_for_training(self, training_points_num = None, train_batch_size = None):
            
        assert training_points_num is not None or self.training_points_num is not None, "Please specify the number of training points"
        training_points_num = training_points_num if training_points_num is not None else self.training_points_num

        assert train_batch_size is not None or self.train_batch_size is not None, "Please specify the batch size"
        train_batch_size = train_batch_size if train_batch_size is not None else self.train_batch_size

        self.train_data_module.set_training_points_num(training_points_num)
        self.train_data_module.set_batch_size(train_batch_size)
        self.train_data_module.reset()
    

    def pretrain(self, epochs=5):
        
        self.model.pretrain = True
        self.model.use_h0 = False
        
        if epochs > 0:
            trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=epochs,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            default_root_dir=self.log_dir,
            reload_dataloaders_every_n_epochs=15,
            accumulate_grad_batches=12,
            )

            torch.autograd.set_detect_anomaly(True)
            trainer.fit(self.model, self.train_data_module.dataloader())

        self.model.set_previous_cbf(self.model.h)



    def train(self, epochs=5):

        self.model.pretrain = False
        self.model.use_h0 = True
        if epochs > 0:
            trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=epochs,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            default_root_dir=self.log_dir,
            reload_dataloaders_every_n_epochs=15,
            accumulate_grad_batches=12,
            )

            torch.autograd.set_detect_anomaly(True)
            trainer.fit(self.model, self.train_data_module.dataloader())


    def prepare_for_testing(self, testing_points_num = None , test_batch_size=None , test_index = None):

        assert test_index is not None or self.test_index is not None, "Please specify the test index"
        test_index = test_index if test_index is not None else self.test_index

        assert testing_points_num is not None or self.testing_points_num is not None, "Please specify the number of testing points"
        testing_points_num = testing_points_num if testing_points_num is not None else self.testing_points_num

        assert test_batch_size is not None or self.test_batch_size is not None, "Please specify the batch size"
        test_batch_size = test_batch_size if test_batch_size is not None else self.test_batch_size
        print( colored(str(testing_points_num), "blue") )
        self.test_data_module.set_testing_points_num(testing_points_num)
        self.test_data_module.set_batch_size(test_batch_size)
        self.test_data_module.set_test_index(test_index)
        self.test_data_module.reset()
       

    def test(self):

        trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=1,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        default_root_dir=self.log_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        )
        trainer.test(self.model, self.test_data_module.dataloader())


    def augment_dataset(self, data_module: VerificationDataModule):

        self.train_data_module.concatenate(data_module)
        