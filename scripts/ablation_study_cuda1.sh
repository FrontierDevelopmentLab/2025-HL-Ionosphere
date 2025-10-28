#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ioncast-physicsnemo

# - JPLD + F10.7 alone
# - JPLD + all of SET
# - JPLD + Ap & Kp
# - JPLD + all of Bx/By/Bz & vx/vy/vz (aka Omniweb) 
# - JPLD + orbital mechanics + Quasidipole
# - JPLD
# - everything but JPLD- but this will be difficult to set up, so skipping for now
# - Train model with all auxiliary datasets on solar minimum (01/01/2018 - 12/31/2020): 2 years is 731 days dilation x, full dataset is 5,194 days dilation 128, so dilation x should be 731/5194 * 128 = 18
# - Train model with all auxiliary datasets on solar maximum (01/01/2013 - 12/31/2015): 2 years is 731 days dilation x, full dataset is 5,194 days dilation 128, so dilation x should be 731/5194 * 128 = 18. Wanted to do 2022-2024, but we only have JPLD data up through August 2024 so using the lower solar max dates instead.
# - Same as full model but without residual targetting

# Training the full model with all auxiliary datasets (reference)- this is the August 14 Showcase model
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon quasidipole celestrak omni set --mode train --target_dir /home/jupyter/halil_debug/ioncastgnn-fulldataset-dilation256_Aug14 --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 1000 --learning_rate 3e-4 --weight_decay 0.0 --context_window 8 --prediction_window 1 --num_evals 1 --jpld_weight 2.0 --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --mesh_level 6 --valid_every_nth_epoch 1 --save_all_models --residual_target --wandb_run_name IonCastGNN --date_dilation 256 --partition_size 1 --device cuda:1 --valid_event_id validation_events_1
declare -a ABLATIONS=(
    "--aux_dataset set --set_columns space_environment_technologies__f107_obs__ space_environment_technologies__f107_average__ --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-f107  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset set --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-set  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset celestrak --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-celestrak  --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    "--aux_dataset omni --omniweb_columns omniweb__bx_gse__[nT] omniweb__by_gse__[nT] omniweb__bz_gse__[nT] omniweb__vx_velocity__[km/s] omniweb__vy_velocity__[km/s] omniweb__vz_velocity__[km/s] --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-omni --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    # "--aux_dataset sunmoon quasidipole --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-qd-sunmoon --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --date_dilation 128 --residual_target"
    # "--target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-noaux --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00  --date_dilation 128 --residual_target"  
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2018-01-01T00:00:00 --date_end 2020-12-31T23:59:59 --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-solarmin --date_dilation 18 --residual_target"  
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2013-01-01T00:00:00 --date_end 2015-12-31T23:59:59 --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-solarmax --date_dilation 18 --residual_target"
    # "--aux_dataset sunmoon quasidipole celestrak omni set --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --target_dir /home/jupyter/linnea_results/ioncastgnn-fulldataset-dilation128-residual-oct22-technical-showcase-ablation-not-residual-target --date_dilation 128"  
)

for i in "${!ABLATIONS[@]}"
do
    echo ">>> Running ablation $((i+1)) / ${#ABLATIONS[@]}: ${ABLATIONS[$i]}"
    CUDA_VISIBLE_DEVICES=1 python run_ioncast.py ${ABLATIONS[$i]} --data_dir /home/jupyter/data --mode train --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 14 --learning_rate 3e-4 --weight_decay 0.0 --context_window 8 --prediction_window 1 --num_evals 1 --jpld_weight 2.0 --mesh_level 6 --device cuda --valid_event_id validation_events_4 --valid_every_nth_epoch 2 --valid_event_seen_id=G0H3-201610140300 --save_all_models --max_valid_samples 1400 --wandb_run_name IonCastGNN_sunlocked_technical_showcase_ablation_$((i+1)) --lead_times 180
done

echo "Ablation study complete"