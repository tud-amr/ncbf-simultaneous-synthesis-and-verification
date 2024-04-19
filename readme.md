

## Install

#### Create a conda environment
```bash
conda create --name <your_env_name> python=3.7

conda activate <your_env_name>
```

#### Install Pytorch
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

#### Install Lightning and StableBaseline
```bash
 pip install lightning==1.9.5 stable-baselines3==1.8.0
```

#### Install other dependencies
```bash
pip install termcolor scipy matplotlib cvxpy cvxpylayers gurobipy gymnasium tensorboard==2.11.2
```

#### Install this repo

```bash
git clone https://github.com/tud-amr/ncbf-simultaneous-synthesis-and-verification.git

cd ncbf-simultaneous-synthesis-and-verification

python -m pip install -e .
```

## Repulicate 

### Inverted Pendulum
0. Configuration

    The configuration files are in *safe_rl_cbf/Configure*

1. Train nCBF

    ```bash
    python3 safe_rl_cbf/main/train_model.py --config_file inverted_pendulum.json
    ```

2. Test nCBF (repulicate Fig. 4 )

    ```bash
    python3 safe_rl_cbf/main/test_model.py --config_file inverted_pendulum.json
    ```

    Figures can be found in *logs/CBF_logs/<prefix>/fig*

3. Safe Learning 

    Training
    ```bash
    python3 safe_rl_cbf/RL/main/train_model.py --config_file inverted_pendulum.json
    ```

    Visualization
    ```bash
    python3 safe_rl_cbf/RL/main/test_model.py --config_file inverted_pendulum.json
    ```

### 2D Navigation

1. train nCBF

    ```bash
    python3 safe_rl_cbf/main/train_model.py --config_file point_robot.json
    ```

2. test nCBF (repulicate Fig. 6d)

    ```bash
    python3 safe_rl_cbf/main/test_model.py --config_file point_robot.json
    ```

3. safe learning (repulicate Fig. 6a and Fig. 6b)

    Training
    ```bash
    python3 safe_rl_cbf/RL/main/train_model.py --config_file point_robot.json
    ```

    Visualization
    ```bash
    python3 safe_rl_cbf/RL/main/test_model.py --config_file point_robot.json
    ```

## Understand Our Code
The structure of the code is similar to our system diagram. see Fig. 

1. BBVT
  
    There is a class named BBVT, which manages the different modules to complete training and verification process. The definition can be found at: 

2. Learner

    Learner store the training and testing data in *DataModule* and optimize neural network through a Pytorch model, named *NeuralCBF*

    *DataModule* is defiened at:

    *NeuralCBF* is defiened at: 

3. Verifier

    Verifier checks if the CBC holds in each hyperrectangles. If not, those hyperrectangles will be refined.

    *Verifier* is defined here: 

    The computation of function Eq.  and Eq. can be found here:

    Hyperrectangle refinement is here:
