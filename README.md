# Control in a continuous action space with DDPG

## Files 

The project contains all the files for the DDPG training with the inverted pendulum environment. 

The implementation of class needed for the DDPG agent can be found in python files (i.e `.py`). 
The testing for each question of the project can be found in jupyter notebook (i.e `ipynb`).

* `buffer.py` : implementation of the Replay Buffer. 
* `networks.py` : implementation of the QNetwork and PolicyNetwork. 
* `helpers.py` : implmentation of the NormalizedEnv and the RandomAgent.
* `heuristic.py` : implementation of the Heuristic agent. 
* `qnetwork.py` : intermediary implementation of the QNetwork and the Replay Buffer, needed for the question 5 of the project. 
* `noise.py` : implentation of the GaussianActionNoise and OUActionNoise.
* `ddpg.py` : implementation of the minimal DDPG algorithm, without the target networks. 
* `ddpg_target.py` : implementation of entire the DDPG algorithm. 
* `test_heuristic.ipynb` : simulation of the Heuristic agent. 
* `test_qnetwork.ipynb` : simulation of the Qnetwork with the Heuristic agent.
* `test_ddpg.ipynb` : simulation of the minimal DDPG agent with the Gaussian noise. 
* `test_ddpg_target.ipynb` : simulation of the DDPG agent with the Gaussian noise. 
* `test_ddpg_noise.ipynb` : simulation of the DDPG agent with the OU noise.

The instructions of the project can be found in the pdf file `Miniproject_DDPG.pdf`. 

Here is the Google Drive link to simulate the notebooks with the Google Colab environment: https://drive.google.com/drive/folders/1eEfZvxUA_WQaQ0dV__J2iqz1DM0FC_UP?usp=share_link


