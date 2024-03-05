from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Models.BBVT import BBVT


############### config file ################
config_file = "inverted_pendulum_test.json"

#################  unpack configurations  #################
config = read_config( os.path.join("safe_rl_cbf/Configure" ,config_file) )

system = select_dynamic_system(config["system"], config["constraints"])

prefix = config["prefix"]
log_dir = config["log_dir"]
load_pretrained = config["load_pretrained"]
pretrained_model_path = config["pretrained_model_path"]

network_structure = config["hyperparameter"]["network_structure"]
training_points_num = config["hyperparameter"]["training_points_num"]
train_batch_size = config["hyperparameter"]["train_batch_size"]
initial_grid_gap = config["hyperparameter"]["initial_grid_gap"]
verify_batch_size = config["hyperparameter"]["verify_batch_size"]
minimum_grid_gap = config["hyperparameter"]["minimum_grid_gap"]
testing_points_num = config["hyperparameter"]["testing_points_num"]
test_batch_size = config["hyperparameter"]["test_batch_size"]
test_index = config["hyperparameter"]["test_index"]
max_epochs = config["hyperparameter"]["max_epochs"]
training_without_verification_epochs = config["hyperparameter"]["training_without_verification_epochs"]
k = config["hyperparameter"]["k"]
learning_rate = config["hyperparameter"]["learning_rate"]
lambda_ = config["hyperparameter"]["lambda_"]

if load_pretrained:
    model = NeuralCBF.load_from_checkpoint(pretrained_model_path, dynamic_system=system, network_structure=network_structure, learning_rate=learning_rate, lambda_=lambda_)
    model.set_previous_cbf(model.h)
   
else:
    model = NeuralCBF(dynamic_system=system, network_structure=network_structure, learning_rate=learning_rate, lambda_=lambda_)


############################ load model ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#################  test  #################
    
bbvt = BBVT(model=model, prefix=config["prefix"] , log_dir=log_dir, 
            training_points_num=training_points_num, train_batch_size=train_batch_size,
            testing_points_num=testing_points_num, test_batch_size=test_batch_size, test_index=test_index,
            initial_grid_gap=initial_grid_gap, verify_batch_size=verify_batch_size, minimum_grip_gap=minimum_grid_gap)

bbvt.prepare_data()

bbvt.test()
bbvt.draw_figures()