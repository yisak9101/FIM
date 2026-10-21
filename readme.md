

# FIM


## Installation instructions

Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda create -n fim python=3.8 -y
conda activate fim

bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:

```shell
bash install_gfootball.sh
```

Update GRF scenarios (Warning: this will overwrite existing map files in the gfootball package):
```shell
python update_grf_scenario.py
```

## Command Line Tool

**Run experiments**

```shell
# Push-2-Box
python main.py --config=FIM_MPE --env-config=push_box with env_args.map_name=push_box alpha=5

# SMAC
python main.py --config=FIM_SMAC --env-config=sc2 with env_args.map_name=3s_vs_5z alpha=10 
python main.py --config=FIM_SMAC --env-config=sc2 with env_args.map_name=corridor alpha=20 
python main.py --config=FIM_SMAC --env-config=sc2 with env_args.map_name=MMM2 alpha=10
python main.py --config=FIM_SMAC --env-config=sc2 with env_args.map_name=6h_vs_8z alpha=20 td_lambda=0.3 epsilon_anneal_time=1000000 
python main.py --config=FIM_SMAC --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z alpha=10 rnn_hidden_dim=256

# GRF
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_2_vs_2 alpha=10
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_3_vs_2 alpha=10
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_4_vs_3 alpha=10
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_counterattack alpha=10
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_2_vs_2_full_field alpha=10 batch_size=128
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_3_vs_2_full_field alpha=10
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_4_vs_3_full_field alpha=10 batch_size=128
python main.py --config=FIM_GRF --env-config=grf with env_args.map_name=academy_counterattack_full_field alpha=10

# SMACv2
pythohn main.py --config=FIM_SMAC --env-config=sc2v2 with env_args.map_name=protoss_5_vs_5 alpha=10
pythohn main.py --config=FIM_SMAC --env-config=sc2v2 with env_args.map_name=terran_5_vs_5 alpha=10
pythohn main.py --config=FIM_SMAC --env-config=sc2v2 with env_args.map_name=zerg_5_vs_5 alpha=10
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


To enable experiment tracking with WandB, append the following parameters:

```shell
wandb=True key={YOUR_API_KEY} project={YOUR_PROJECT} entity={YOUR_ENTITY}
```

All results will be stored in the `Results` folder and named with `map_name`, and we store the test wining rate with csv format in `csv_files`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```
