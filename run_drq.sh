#############################
########## CRESP-T ##########
#############################
task_name=(cartpole_swingup cartpole_swingup cartpole_swingup cartpole_swingup cartpole_swingup cartpole_swingup)
cuda_id=(2 3 4 1 2 7)
seed=(1 2 3 4 5 6)
dcs_type=color
num_sources=1
num_train_frames=500101
rew_noise=true
rew_noise_std=0.1
rew_noise_rate=1.0
dynamic_noise=false
dynamic_noise_std=0.1
dynamic_noise_rate=1.0

for ((i=0;i<${#cuda_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${cuda_id[i]} nohup python -u VRL/train.py \
        task_name=${task_name[i]} \
        task=online_dmc \
        num_train_frames=${num_train_frames} \
        batch_size=64 \
        lr=0.001 \
        critic_update_times=2 \
        algo=drq \
        feature_dim=64 \
        rew_noise.enable=${rew_noise} \
        rew_noise.std=${rew_noise_std} \
        rew_noise.rate=${rew_noise_rate} \
        dynamic_noise.enable=${dynamic_noise} \
        dynamic_noise.eval_enable=${dynamic_noise} \
        dynamic_noise.std=${dynamic_noise_std} \
        dynamic_noise.rate=${dynamic_noise_rate} \
        eval_on_dcs=true \
        num_sources=${num_sources} \
        dcs_type=${dcs_type} \
        data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
        use_tb=true \
        use_wandb=true \
        seed=${seed[i]} \
        exp_name=drq-noise-${dcs_type}${num_sources} \
            > ./log/crespt/drq-rew_noise-${dcs_type}${num_sources}-${task_name[i]}-s${seed[i]}.log 2>&1 &
done
