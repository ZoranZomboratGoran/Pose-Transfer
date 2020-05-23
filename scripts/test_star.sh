# Market1501
python test.py --dataroot ./market_data/ --name market_PATN_v1.0 --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 2 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./market_data/market-pairs-test.csv --which_epoch latest --results_dir ./results_v1.0

python3 test.py --dataroot ./market_data/ --name market_PATN --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 2 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --which_epoch latest --results_dir ./results

# DeepFashion
python test.py --dataroot ./fashion_data/ --name fashion_PATN_v1.0 --model PATN --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results_v1.0


