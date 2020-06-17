# AdMRL

This is the implementation for the paper Model-based Adversarial Meta-Reinforcement Learning (https://arxiv.org/abs/2006.08875). 

## Requirements
1. OpenAI Baselines (0.1.6)
2. MuJoCo (>= 1.5)
3. TensorFlow (>= 1.9)
4. NumPy (>= 1.14.5)
5. Python 3.6

## Installment
```bash
pip install -r requirements.txt
cd rllab
pip install -e .
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
setup mujoco with ./scripts/setup_mujoco.sh
export MUJOCO_LICENSE_PATH=<path of mjkey.txt>
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -r requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path of mujoco200/bin>
pip install mujoco_py==2.0.2.5
```

## Run
You can specify the hyper-parameter in launch.py and run experiments:
```bash
python launch.py
```

## License

MIT License.
