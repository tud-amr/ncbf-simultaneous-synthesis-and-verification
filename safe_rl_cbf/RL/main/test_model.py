from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Models.BBVT import BBVT


parser = argparse.ArgumentParser(description='Train a neural network')

parser.add_argument('--config_file', type=str, default="inverted_pendulum.json", help='please type the config file name in folder safe_rl_cbf/Configure')

args = parser.parse_args()

############### config file ################
config_file = args.config_file

#################  unpack configurations  #################
config = read_config( os.path.join("safe_rl_cbf/Configure" ,config_file) )

system = select_dynamic_system(config["system"], config["constraints"])

prefix = config["prefix"]
log_dir = config["log_dir"]
network_structure = config["network_structure"]
gamma = config["gamma"]

model_path = config["test"]["model_path"]
testing_points_num = config["test"]["hyperparameter"]["testing_points_num"]
test_batch_size = config["test"]["hyperparameter"]["test_batch_size"]
test_index = config["test"]["hyperparameter"]["test_index"]

model = NeuralCBF.load_from_checkpoint(model_path, dynamic_system=system, network_structure=network_structure, gamma=gamma)

############################ load model ###################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#################  test  #################
    
bbvt = BBVT(model=model, prefix=prefix , log_dir=log_dir, 
            testing_points_num=testing_points_num, test_batch_size=test_batch_size, test_index=test_index,
            )

bbvt.prepare_data()

bbvt.test()
bbvt.draw_figures()