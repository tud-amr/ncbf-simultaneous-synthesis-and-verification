from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

current_date_str = datetime.datetime.now().strftime("%d_%b")

########################### hyperparameters #############################

train_mode = 0
system = inverted_pendulum_1
default_root_dir = "logs/CBF_logs/IP_" + current_date_str
checkpoint_dir = "saved_models/inverted_pendulum_stage_1/checkpoints/epoch=293-step=2646.ckpt"
grid_gap = torch.Tensor([0.2, 0.2])  

########################## start training ###############################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, save_top_k=1, monitor="Total_loss/train")
if train_mode==0:

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(5e5), train_mode=train_mode)

    # NN = NeuralCBF(dynamic_system=system, data_module=data_module, train_mode=train_mode)
    # NN0 =  NeualCBF.load_from_checkpoint("logs/CBF_logs/dubins_car_acc/lightning_logs/version_1/checkpoints/epoch=86-step=14181.ckpt",dynamic_system=system, data_module=data_module, require_grad_descent_loss=True, primal_learning_rate=8e-4, fine_tune=fine_tune)
    NN = NeuralCBF.load_from_checkpoint(checkpoint_dir, dynamic_system=system, data_module=data_module, train_mode=train_mode)
   
    NN.training_stage = 0
    NN.set_previous_cbf(NN.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=20,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        # gradient_clip_val=1
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)

elif train_mode==1:
        
    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(1e5), train_mode=train_mode)

    NN0 =  NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
    NN = NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
    # NN = NeualCBF(dynamic_system=system, data_module=data_module, train_mode=train_mode)

    # NN.set_previous_cbf(NN0.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=200,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=15,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)
    
elif train_mode==2:
     
    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=256, training_points_num=int(1e6), train_mode=1, training_grid_gap=None)

    NN0 =  NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
    NN = NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
   
    # NN.set_previous_cbf(NN0.h)

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=5,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=1000,
        accumulate_grad_batches=24,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)
    
    current_learning_rate = NN.learning_rate
    del NN, trainer
    
    verification_time = 0
    training_start_time = time.time()

    for i in range(20):
        print(f"the iteration {i} of verification")
        verification_start_time = time.time()
        
        data_module.train_mode = 2
        data_module.training_grid_gap = grid_gap
        data_module.prepare_data()

        log_dir = default_root_dir + "/lightning_logs"
        version_list = os.listdir(log_dir)
        version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
        checkpoint_folder_dir = version_dir + "/checkpoints"
        checkpoint_name = os.listdir(checkpoint_folder_dir)[0]
        latest_checkpoint = checkpoint_folder_dir + "/" + checkpoint_name 
        print("latest_checkpoint: ", latest_checkpoint)

        NN0 =  NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
        NN = NeualCBF.load_from_checkpoint(latest_checkpoint,dynamic_system=system, data_module=data_module, train_mode=train_mode)
    
        # NN.set_previous_cbf(NN0.h)

        trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=1000,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
            default_root_dir=default_root_dir,
            reload_dataloaders_every_n_epochs=1,
            accumulate_grad_batches=12,
            # gradient_clip_val=0.5
            )

        torch.autograd.set_detect_anomaly(True)
        trainer.fit(NN)

        print(f"data_module.verified = {data_module.verified}")
        print(f"augment_data.shape = {data_module.augment_data.shape}")
        # torch.save(data_module.augment_data, "s_training.pt")
       
        verification_time += time.time() - verification_start_time
        counter_examples_num = data_module.augment_data.shape[0]

        if data_module.verified == 1:
            print(f"Verification time: {verification_time}")
            print(f"counter_examples_num: {counter_examples_num}")
            break
        
        del NN, trainer

        data_module.train_mode = 1
        data_module.training_grid_gap = None
        data_module.augment_dataset()
        num_of_augment_data = data_module.augment_data.shape[0]
        

        log_dir = default_root_dir + "/lightning_logs"
        version_list = os.listdir(log_dir)
        version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
        checkpoint_folder_dir = version_dir + "/checkpoints"
        checkpoint_name = os.listdir(checkpoint_folder_dir)[0]
        latest_checkpoint = checkpoint_folder_dir + "/" + checkpoint_name 

        NN0 =  NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=1)
        NN = NeualCBF.load_from_checkpoint(latest_checkpoint,dynamic_system=system, data_module=data_module, train_mode=1)
    
        # NN.set_previous_cbf(NN0.h)

        trainer = pl.Trainer(
            accelerator = "gpu",
            devices = 1,
            max_epochs=20,
            # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
            # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
            default_root_dir=default_root_dir,
            reload_dataloaders_every_n_epochs=1000,
            accumulate_grad_batches=24,
            # gradient_clip_val=0.5
            )

        torch.autograd.set_detect_anomaly(True)
        trainer.fit(NN)

        current_learning_rate = NN.learning_rate

elif train_mode==3:

    data_module = TrainingDataModule(system=system, val_split=0, train_batch_size=1024, training_points_num=int(1e4), train_mode=3, training_grid_gap=None)

    NN0 =  NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
    NN = NeualCBF.load_from_checkpoint(checkpoint_dir,dynamic_system=system, data_module=data_module, train_mode=train_mode)
   
    NN.set_previous_cbf(NN0.h)
    data_module.model = NN

    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = 1,
        max_epochs=31,
        # callbacks=[ EarlyStopping(monitor="Total_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=20, stopping_threshold=1e-3) ], 
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        default_root_dir=default_root_dir,
        reload_dataloaders_every_n_epochs=10,
        accumulate_grad_batches=12,
        # gradient_clip_val=0.5
        )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(NN)

    print(f"data_module.verified = {data_module.verified}")
    print(f"augment_data.shape = {data_module.augment_data.shape}")
    print(f"verification time = {data_module.SMT_verification_time}")
    print(f"average generation time each counterexample = {data_module.SMT_verification_time/data_module.SMT_CE_num}")







    # log_dir = default_root_dir + "/lightning_logs"

    # version_list = os.listdir(log_dir)

    # version_dir =  log_dir + "/" +  max(version_list,key=extract_number)
    

    # checkpoint_folder_dir = version_dir + "/checkpoints"
    

    # checkpoint_name = os.listdir(checkpoint_folder_dir)[0]

    # checkpoint_path = checkpoint_folder_dir + "/" + checkpoint_name 
    

    # NN = NeualCBF.load_from_checkpoint(checkpoint_path,dynamic_system=inverted_pendulum_1, data_module=data_module, require_grad_descent_loss=True, fine_tune=fine_tune)

    # trainer = pl.Trainer(
    #     accelerator = "gpu",
    #     devices = 1,
    #     max_epochs=200,
    #     # callbacks=[ EarlyStopping(monitor="Safety_loss/train", mode="min", check_on_train_epoch_end=True, strict=False, patience=50, stopping_threshold=1e-3) ], 
    #     # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
    #     default_root_dir=default_root_dir,
    #     reload_dataloaders_every_n_epochs=15,
    #     accumulate_grad_batches=12,
    #     )

    # torch.autograd.set_detect_anomaly(True)
    # # trainer.fit(NN, ckpt_path="/home/wangxinyu/.mujoco/mujoco210/sunny_test/masterthesis_test/CBF_logs/robust_training_maximum/lightning_logs/version_4/checkpoints/epoch=92-step=1116.ckpt")
    # trainer.fit(NN)

    # torch.save(NN.data_module.s_training, "s_training.pt")
