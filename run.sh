#############################
########## CRESP-T ##########
#############################
CUDA_VISIBLE_DEVICES=5 nohup python -u VRL/train.py \
    task_name=walker_walk \
    task=online_dmc \
    num_train_frames=500001 \
    batch_size=128 \
    lr=0.001 \
    critic_update_times=2 \
    algo=crespt \
    use_cf=True \
    feature_dim=64 \
    agent.rsd_nstep=2 \
    agent.cf_temp=0.01 \
    agent.cf_update_freq=1 \
    agent.omega_num_sample=32 \
    agent.omega_cf_output_mode=min_mu \
    eval_on_dcs=True \
    num_sources=3 \
    dcs_type=background \
    data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
    use_tb=False \
    seed=0 \
    exp_name=crespt-rsd2-background3 > ./log/crespt/crespt-rsd2-background3-ww-s0.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python -u VRL/train.py \
#     task_name=walker_walk \
#     task=online_dmc \
#     num_train_frames=500001 \
#     batch_size=128 \
#     lr=0.001 \
#     critic_update_times=2 \
#     algo=crespt \
#     use_cf=True \
#     feature_dim=64 \
#     agent.rsd_nstep=2 \
#     agent.cf_temp=0.01 \
#     agent.cf_update_freq=1 \
#     agent.omega_num_sample=32 \
#     agent.omega_cf_output_mode=min_mu \
#     eval_on_dcs=True \
#     scale=0.1 \
#     scale_end=0.2 \
#     num_sources=3 \
#     dcs_type=color \
#     data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
#     use_tb=False \
#     seed=0 \
#     exp_name=crespt-rsd2-color3_10_20 > ./log/crespt/crespt-rsd2-color3_10_20-ww-s0.log 2>&1 &


###########################
########## CRESP ##########
###########################
# CUDA_VISIBLE_DEVICES=3 nohup python -u VRL/train.py \
#     task_name=walker_walk \
#     task=online_dmc \
#     num_train_frames=500001 \
#     batch_size=128 \
#     lr=0.001 \
#     critic_update_times=2 \
#     algo=cresp \
#     use_cf=True \
#     feature_dim=64 \
#     agent.rsd_nstep=5 \
#     agent.cf_temp=0.01 \
#     agent.cf_update_freq=1 \
#     agent.omega_num_sample=32 \
#     agent.omega_cf_output_mode=min_mu \
#     eval_on_dcs=True \
#     num_sources=3 \
#     dcs_type=background \
#     data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
#     use_tb=False \
#     seed=0 \
#     exp_name=cresp-rsd5-background3 > ./log/cresp/cresp-rsd5-background3-ww-s0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python -u VRL/train.py \
#     cuda_id=5 \
#     task_name=walker_walk \
#     task=online_dmc \
#     num_train_frames=500001 \
#     batch_size=128 \
#     lr=0.001 \
#     critic_update_times=2 \
#     algo=cresp \
#     use_cf=True \
#     feature_dim=64 \
#     agent.rsd_nstep=5 \
#     agent.cf_temp=0.01 \
#     agent.cf_update_freq=1 \
#     agent.omega_num_sample=32 \
#     agent.omega_cf_output_mode=min_mu \
#     eval_on_dcs=True \
#     scale=0.1 \
#     scale_end=0.2 \
#     num_sources=3 \
#     dcs_type=color \
#     data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
#     use_tb=False \
#     seed=0 \
#     exp_name=cresp-rsd5-color2_10_20 > ./log/cresp/cresp-rsd5-color2_10_20-ww-s0.log 2>&1 &
