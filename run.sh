#############################
########## CRESP-T ##########
#############################
task_name=(cartpole_swingup cartpole_swingup cartpole_swingup cheetah_run cheetah_run cheetah_run)
cuda_id=(2 2 4 4 2 5)
seed=(4 5 6 4 5 6)
dcs_type=background
num_sources=2
num_train_frames=600001
cf_lr=0.001
update_crest_times=1

for ((i=0;i<${#cuda_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${cuda_id[i]} nohup python -u VRL/train.py \
        task_name=${task_name[i]} \
        task=online_dmc \
        num_train_frames=${num_train_frames} \
        batch_size=128 \
        lr=0.001 \
        cf_lr=${cf_lr} \
        critic_update_times=2 \
        update_crest_times=${update_crest_times} \
        algo=crespt \
        use_cf=true \
        feature_dim=64 \
        rew_noise.enable=true \
        agent.rsd_nstep=2 \
        agent.cf_temp=0.01 \
        agent.cf_update_freq=1 \
        agent.omega_num_sample=32 \
        agent.omega_cf_output_mode=min_mu \
        eval_on_dcs=true \
        num_sources=${num_sources} \
        dcs_type=${dcs_type} \
        data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
        use_tb=true \
        seed=${seed[i]} \
        exp_name=crespt-rsd2-noise-${dcs_type}${num_sources} \
            > ./log/crespt/crespt-noise-${dcs_type}${num_sources}-${task_name[i]}-s${seed[i]}.log 2>&1 &
done


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
