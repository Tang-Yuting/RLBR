# RL from Bagged Reward

This repository contains the Jax implementation of RL from Bagged Reward.


## Docker

```bash
cd docker
docker build -t docker . -f Dockerfile 
```

### Tips
```bash
# If you have problems with Cython, you can try:
pip uninstall Cython
pip install Cython==3.0.0a10
```



## Example

### Proposed Method

```bash
# Fixed-length reward bags
CUDA_VISIBLE_DEVICES=${device_num} python -m examples.train_reward_model --env_name=${env_name} --save_dir=./tmp_result/ --bag_len=${bag_len} --seed=${seed}

# Aarbitrary reward bags
CUDA_VISIBLE_DEVICES=${device_num} python -m examples.train_arbitrary_reward_model --env_name=${env_name} --save_dir=./tmp_result_arbitrary/ --seed=${seed}"
```

