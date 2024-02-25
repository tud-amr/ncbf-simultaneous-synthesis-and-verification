from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Models.BBVT import BBVT


############### config file ################
config_file = "test.json"
config = read_config( os.path.join("safe_rl_cbf/Configure" ,config_file) )


#################  unpack configurations  #################
train_mode = 1
system = select_dynamic_system(config["system"])
prefix = config["prefix"]
log_dir = config["log_dir"]
load_pretrained = config["load_pretrained"]
pretrained_model_path = config["pretrained_model_path"]

training_points_num = config["hyperparameter"]["training_points_num"]
train_batch_size = config["hyperparameter"]["train_batch_size"]
initial_grid_gap = config["hyperparameter"]["initial_grid_gap"]
verify_batch_size = config["hyperparameter"]["verify_batch_size"]
testing_points_num = config["hyperparameter"]["testing_points_num"]
test_batch_size = config["hyperparameter"]["test_batch_size"]
test_index = config["hyperparameter"]["test_index"]
max_epochs = config["hyperparameter"]["max_epochs"]
k = config["hyperparameter"]["k"]

if load_pretrained:
    model = NeuralCBF.load_from_checkpoint(pretrained_model_path, dynamic_system=system, train_mode=train_mode)
    model.set_previous_cbf(model.h)
else:
    model = NeuralCBF(dynamic_system=system, train_mode=train_mode)


#################  train and verify  #################
    
bbvt = BBVT(model=model, prefix=config["prefix"] , log_dir=log_dir, training_points_num=training_points_num, train_batch_size=train_batch_size,
            testing_points_num=testing_points_num, test_batch_size=test_batch_size, test_index=test_index,
            initial_grid_gap=initial_grid_gap, verify_batch_size=verify_batch_size)

bbvt.prepare_data()
bbvt.training_and_verifying(max_epochs=max_epochs, k=k)

bbvt.test()
bbvt.draw_figures()