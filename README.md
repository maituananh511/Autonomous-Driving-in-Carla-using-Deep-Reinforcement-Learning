# <p align="center"> Autonomous Driving in CARLA using Deep Reinforcement Learning </p>

Artificial Intelligence (AI) is growing extraordinarily in almost every area of technology, and research into self-driving cars is one of them. In this project, we utilize state-of-the-art Deep Reinforcement Learning (DRL) methods to train an agent to drive autonomously. We use the open-source simulator [CARLA](http://carla.org/), which provides a hyper-realistic urban simulation environment for training and evaluating our models — since raw RL algorithms come with real risks and safety concerns, simulation lets us test them without those consequences.

This project implements **two DRL agents**, each trained to navigate a predetermined route in a simulated urban environment:

1. **Proximal Policy Optimization (PPO)** — an on-policy algorithm operating over a continuous state and action space.
2. **Dueling Deep Q-Network (Dueling DQN)** — an off-policy, value-based algorithm operating over a discretized action space.

Both agents share a common perception pipeline built around a **Variational Autoencoder (VAE)**, which compresses high-dimensional visual observations into a compact latent space, allowing both agents to learn faster and more reliably.

This project builds on the original single-agent (PPO) implementation by Idrees Razak ([repo](https://github.com/idreesshaikh)), extending it with a complete, trained Dueling DQN agent alongside the existing PPO agent.

## About the Project

This work aims to develop an end-to-end solution for autonomous driving that sends control commands to the vehicle to help it stay on route and avoid collisions. It is divided into the following components:

1. CARLA environment setup.
2. Variational Autoencoder (shared perception module).
3. Proximal Policy Optimization (continuous agent).
4. Dueling Deep Q-Network (discrete agent).

We use [CARLA](http://carla.org/) (version 0.9.8) as our environment (urban simulator).

### Prerequisites

We're using [CARLA](https://github.com/carla-simulator/carla/releases) (0.9.8) + Additional Maps. This project is mainly focused on two towns — Town 02 and Town 07 — so it's recommended to download the Additional Maps package alongside the CARLA server, and copy the maps from the **Additional Maps** directory into the **Main** CARLA directory.

CARLA supports **Windows** and **Linux**, so it's recommended to set up the project on one of these two OSs.

## Project Setup (Installation)

1. Clone this repository.
2. Make sure you have **Python 3.7+ (64-bit)** installed.
3. Create a virtual environment: `python -m venv venv`.
4. Activate it: `source venv/Scripts/activate` (Windows) or `source venv/bin/activate` (Linux).
5. Install dependencies with pip: `pip install -r requirements.txt`.
6. This project also uses **Poetry** for some dependencies: `cd poetry/ && poetry update`.
7. Download the **CARLA server (0.9.8)** + **Additional Maps** per the Prerequisites above.

Once the CARLA server is running, start a client, e.g.:

```
python continuous_driver.py --exp-name=ppo --train=False
```

## Built With

* [Python](https://www.python.org/downloads/release/python-370/) — Programming language
* [PyTorch](https://pytorch.org/) — Open source machine learning framework
* [CARLA](http://carla.org/) — Urban driving simulator
* [Poetry](https://python-poetry.org/) — Packaging and dependency manager
* [Tensorboard](https://www.tensorflow.org/tensorboard) — Visualization toolkit

# Methodology

The architecture centers on four essential components:

1. CARLA simulation.
2. Shared VAE encoder.
3. PPO agent (continuous control).
4. Dueling DQN agent (discrete control).

## How to Run

### Running a Trained PPO Agent

Pretrained PPO agents are provided for both towns (Town 02 & Town 07), stored in `preTrained_models/PPO/<town>/`.

```
python continuous_driver.py --exp-name ppo --train False
```

By default this runs on Town 07. To use Town 02:

```
python continuous_driver.py --exp-name ppo --train False --town Town02
```

### Running a Trained Dueling DQN Agent

Pretrained Dueling DQN agents are provided for both towns, stored in `preTrained_models/DQN/<town>/`.

```
python discrete_driver.py --exp-name dqn --train False
```

To use Town 02:

```
python discrete_driver.py --exp-name dqn --train False --town Town02
```

### Training a New Agent

To train a new PPO agent:

```
python continuous_driver.py --exp-name ppo
```

To train a new Dueling DQN agent:

```
python discrete_driver.py --exp-name dqn
```

Checkpoints are written to `checkpoints/<algorithm>/<town>/`, and metrics are logged to `logs/<algorithm>/<town>/`. By default training runs on Town 07; add `--town Town02` to train on Town 02 instead.

## Variational Autoencoder

The VAE is trained by driving around the environment (both automatically and manually) to collect semantically segmented images, which are then used as input to the autoencoder. The VAE's weights are frozen once trained, so both the PPO and DQN agents train on top of a fixed, shared latent representation.

To check reconstructed images from a trained VAE:

```
cd autoencoder && python reconstructor.py
```

## Project Architecture Pipeline

Both agents follow the same encode → decide → act pipeline: raw camera observations are compressed by the shared VAE encoder into a low-dimensional latent vector, which is then fed into the respective agent (PPO or Dueling DQN) to produce a driving action.

# File Overview

| File                          | Description                                                                                                           |
| ------------------------------| --------------------------------------------------------------------------------------------------------------------- |
| continuous_driver.py          | Script for training/testing the continuous agent (PPO)                                                                |
| discrete_driver.py            | Script for training/testing the discrete agent (Dueling DQN)                                                          |
| encoder_init.py               | Uses the trained VAE encoder to turn incoming images (states) into latent space                                       |
| parameters.py                 | Contains the hyperparameters of the project                                                                           |
| simulation/connection.py      | CARLA environment class that makes the connection with the CARLA server                                               |
| simulation/environment.py     | CARLA environment class with the core environment setup (gym-inspired class structure)                                |
| simulation/sensors.py         | CARLA environment file containing the agent's sensor classes (setup)                                                  |
| simulation/settings.py        | CARLA environment file containing environment setup parameters                                                        |
| runs/                         | Folder containing Tensorboard plots/graphs                                                                            |
| preTrained_models/PPO         | Folder containing pretrained PPO models' serialized files                                                             |
| preTrained_models/DQN         | Folder containing pretrained Dueling DQN models' serialized files                                                     |
| networks/on_policy/agent.py   | Contains the PPO agent code                                                                                           |
| networks/on_policy/ppo.py     | Contains the PPO network code                                                                                         |
| networks/off_policy/agent.py  | Contains the Dueling DQN agent code                                                                                    |
| networks/off_policy/dqn.py    | Contains the Dueling DQN network code                                                                                  |
| logs/                         | Folder containing logged training metrics for both agents                                                             |
| info/                         | Folder containing figures, gifs, diagrams, & documentation for the project                                            |
| checkpoints/                  | Folder containing serialized agent parameters saved during training                                                   |
| carla/                        | Folder containing the CARLA egg file used to connect to the server                                                    |
| autoencoder/                  | Folder containing the code for the shared Variational Autoencoder (VAE)                                               |

## Viewing Training Progress in Tensorboard

```
tensorboard --logdir runs/
```

## Author

**Mai Tuan Anh** — [GitHub](https://github.com/maituananh511)

## License

This project is licensed under the Apache-2.0 License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds on the original PPO-only implementation by **Idrees Razak** — [GitHub](https://github.com/idreesshaikh), [LinkedIn](https://www.linkedin.com/in/idreesrazak/) — extending it with a complete Dueling DQN agent.
