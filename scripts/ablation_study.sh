#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ioncast-physicsnemo-halil

# - JPLD + F10.7 alone
# - JPLD + all of SET
# - JPLD + Ap & Kp
# - JPLD + all of Bx/By/Bz & vx/vy/vz (aka Omniweb)
# - JPLD + orbital mechanics + Quasidipole
# - JPLD
# - everything but JPLD- but this will be difficult to set up, so skipping for now
# - Train model with all auxiliary datasets on solar minimum (01/01/2018 - 12/31/2020): 2 years is 731 days dilation x, full dataset is 5,194 days dilation 128, so dilation x should be 731/5194 * 128 = 18
# - Train model with all auxiliary datasets on high solar maximum (01/01/2022 - 12/31/2024): 2 years is 731 days dilation x, full dataset is 5,194 days dilation 128, so dilation x should be 731/5194 * 128 = 18
# - Same as full model but without residual targetting

# Training the full model with all auxiliary datasets (reference)
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon quasidipole celestrak omni set --mode train --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 1000 --learning_rate 3e-5 --weight_decay 1e-3 --context_window 8 --prediction_window 1 --num_evals 1 --jpld_weight 2.0 --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --mesh_level 6 --device cuda:0 --valid_event_id validation_events_4 --valid_every_nth_epoch 2 --valid_event_seen_id=G0H3-201610140300 --save_all_models --residual_target --max_valid_samples 1400 --wandb_run_name IonCastGNN_sunlocked_residual_technical_showcase --sunlock_features --date_dilation 128 --lead_times 180


declare -a ABLATIONS=(
    "--aux_dataset set --set_columns space_environment_technologies__f107_obs__ space_environment_technologies__f107_average__ --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-f107  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset set --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-set  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset celestrak --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-celestrak  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset omni --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-omni --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset sunmoon quasidipole --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-qd-sunmoon --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-noaux --date_dilation 128 --residual_target"  
)
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2018-01-01T00:00:00 --date_end 2020-12-31T23:59:59 --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-solarmin --date_dilation 18 --residual_target"  
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2022-01-01T00:00:00 --date_end 2024-08-01T00:00:00 --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-solarmax --date_dilation 18 --residual_target"
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-sunlock-dilation128-residual-oct22-technical-showcase-ablation-not-residual-target --date_dilation 128"  

for i in "${!ABLATIONS[@]}"
do
    echo ">>> Running ablation $((i+1)) / ${#ABLATIONS[@]}: ${ABLATIONS[$i]}"
    CUDA_VISIBLE_DEVICES=1 python run_ioncast.py ${ABLATIONS[$i]} --data_dir /home/jupyter/data --mode train --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 14 --learning_rate 3e-5 --weight_decay 1e-3 --context_window 8 --prediction_window 1 --num_evals 1 --jpld_weight 2.0 --mesh_level 6 --device cuda --valid_event_id validation_events_4 --valid_every_nth_epoch 2 --valid_event_seen_id=G0H3-201610140300 --save_all_models --max_valid_samples 1400 --wandb_run_name IonCastGNN_sunlocked_technical_showcase_ablation_$((i+1)) --sunlock_features --lead_times 180
done

echo "Ablation study complete"