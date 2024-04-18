from safe_rl_cbf.Models.common_header import *
from safe_rl_cbf.Models.custom_header import *
from safe_rl_cbf.Models.BBVT import BBVT

parser = argparse.ArgumentParser(description='RL training')

parser.add_argument('--config_file', type=str, default="inverted_pendulum.json", help='please type the config file name in folder safe_rl_cbf/Configure')

args = parser.parse_args()

############### config file ################
config_file = args.config_file

#################  unpack configurations  #################
config = read_config( os.path.join("safe_rl_cbf/Configure" ,config_file) )

system = select_dynamic_system(config["system"], config["constraints"])
rl_env, _ = select_RL_env(config["system"])

prefix = config["prefix"]
log_dir = config["log_dir"]
network_structure = config["network_structure"]
gamma = config["gamma"]

cbf_model_dir = config["RL"]["cbf_model_path"]
rl_model_dir = config["RL"]["rl_model_path"]


cbf_model = NeuralCBF.load_from_checkpoint(cbf_model_dir, dynamic_system=system, network_structure=network_structure, gamma=gamma)

env = rl_env(render_sim=False)
env.set_barrier_function(cbf_model)

model = PPO.load(rl_model_dir)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if  dones is True:     
        obs = env.reset()