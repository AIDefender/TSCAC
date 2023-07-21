# Two-Stage Constrained Actor-Critic for Short Video Recommendation (TSCAC)

## Installation

```bash
pip install -r requirements.txt
```

## Training TSCAC

### Preparing Training Data
1. Download the [KuaiRand](https://kuairand.com/) dataset.
2. Specify custome data folder in line 8 of `process.py`.
3. Run `python process.py`.


### Performing Stage One Training
Run `main.py` with the following config:
```python
python main.py --config=multi_critic_krand with save_model=True bc_load_path=krand_sl_onehot_eval seed=1 behavior_onehot=True exp_name=rcpo_full
```

### Performing Stage Two Training
Run `main.py` with the following config:
```python
python main.py --config=multi_critic_awac_ddpg_krand with save_model=True bc_load_path=krand_sl_onehot_eval seed=0 behavior_onehot=True exp_name=full_sigma30_k0_0001_new_ratio_seed0 kl_loss_coef=0.0001 sigma=30 constrained_policy_model_path=/results/multi_critic_krand/rcpo_full/1/models/ new_ratio=True
```
where `constrained_policy_model_path` is the checkpoint path of the policy trained in stage one.

If you find our code/paper useful, please consider citing our work:
```bib
@inproceedings{DBLP:conf/www/0001XZX0ZWZXZJG23,
  author       = {Qingpeng Cai and
                  Zhenghai Xue and
                  Chi Zhang and
                  Wanqi Xue and
                  Shuchang Liu and
                  Ruohan Zhan and
                  Xueliang Wang and
                  Tianyou Zuo and
                  Wentao Xie and
                  Dong Zheng and
                  Peng Jiang and
                  Kun Gai},
  title        = {Two-Stage Constrained Actor-Critic for Short Video Recommendation},
  booktitle    = {{WWW}},
  pages        = {865--875},
  publisher    = {{ACM}},
  year         = {2023}
}
```

Feel free to reach out if you have any questions!