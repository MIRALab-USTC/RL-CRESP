#############################
########## CRESP-T ##########
#############################
task_name=(cartpole_swingup cartpole_swingup cartpole_swingup cheetah_run cheetah_run cheetah_run)
cuda_id=(4 5 6 7 4 5)
seed=(4 5 6 4 5 6)
dcs_type=background
num_sources=2
num_train_frames=600001

for ((i=0;i<${#cuda_id[@]};i++))
do
    CUDA_VISIBLE_DEVICES=${cuda_id[i]} nohup python -u VRL/train.py \
        task_name=${task_name[i]} \
        task=online_dmc \
        num_train_frames=${num_train_frames} \
        batch_size=128 \
        lr=0.001 \
        critic_update_times=2 \
        algo=drq \
        feature_dim=64 \
        rew_noise.enable=true \
        eval_on_dcs=true \
        num_sources=${num_sources} \
        dcs_type=${dcs_type} \
        data_path=/home/amax/sda6/amax/rl/data/DAVIS/JPEGImages/480p/ \
        use_tb=true \
        seed=${seed[i]} \
        exp_name=drq-noise-${dcs_type}${num_sources} \
            > ./log/crespt/drq-noise-${dcs_type}${num_sources}-${task_name[i]}-s${seed[i]}.log 2>&1 &
done
