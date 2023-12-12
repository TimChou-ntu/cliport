export CLIPORT_ROOT=$(pwd)

python -m pdb cliport/rl_with_env.py model_task=multi-language-conditioned \
                       PPO_task=packing-boxes-pairs-seen-colors \
                       agent=cliport \
                       mode=train \
                       n_demos=100 \
                       train_demos=1000 \
                       exp_folder=cliport_quickstart \
                       update_results=True \
                       disp=False \
                       model_path=/mnt/sdb/timothy/Desktop/2023Fall/cliport/cliport_quickstart/multi-language-conditioned-cliport-n1000-train/checkpoints/