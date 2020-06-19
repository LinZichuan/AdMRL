# AdMRL

This is the implementation for the paper [Model-based Adversarial Meta-Reinforcement Learning](https://arxiv.org/abs/2006.08875). 

## Requirements
1. OpenAI Baselines (0.1.6)
2. MuJoCo (>= 1.5)
3. TensorFlow (>= 1.9)
4. NumPy (>= 1.14.5)
5. Python 3.6

## 🔧 Installation
To install, you need to first install [MuJoCo](https://www.roboti.us/index.html). Set `LD_LIBRARY_PATH` to point to the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) and `MUJOCO_LICENSE_PATH` to point to the MuJoCo license (`/$HOME/.mujoco/mjkey.txt`). You can then setup mujoco by running `rllab/scripts/setup_mujoco.sh`.
To install the remaining dependencies, you can create our environment with `conda env create -f environment.yml`. To use rllab, you also need to run `cd rllab; pip install -e .`.

## 🚀 Run
You can run experiments:
```bash
python main.py --taskname=Ant2D
```
You can also specify the hyper-parameters in launch.py and run many experiments:
```bash
python launch.py
```

## License

MIT License.
