from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Models.BBVT import BBVT

parser = argparse.ArgumentParser(description='Train a neural network')

parser.add_argument('--config_file', type=str, default="inverted_pendulum_pretrained.json", help='please type the config file name in folder safe_rl_cbf/Configure')

args = parser.parse_args()

############### config file ################
config_file = args.config_file

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

#################  train and verify  #################
    
bbvt = BBVT(model=model, prefix=config["prefix"] , log_dir=log_dir, 
            training_points_num=training_points_num, train_batch_size=train_batch_size,
            testing_points_num=testing_points_num, test_batch_size=test_batch_size, test_index=test_index,
            initial_grid_gap=initial_grid_gap, verify_batch_size=verify_batch_size, minimum_grip_gap=minimum_grid_gap)

bbvt.prepare_data()
bbvt.training_and_verifying(max_epochs=max_epochs, 
                            training_without_verification_epochs=training_without_verification_epochs,
                            k=k)

bbvt.test()
bbvt.draw_figures()