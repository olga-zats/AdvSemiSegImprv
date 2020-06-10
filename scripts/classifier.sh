export CUDA_VISIBLE_DEVICES=0
cd ..
python train_classifier_branch_only_VOCcls.py --seed 0 --partial-data 0.125 --save-pred-every 1000 --num-steps 5000 --snapshot-dir snapshots/classifier --batch-size 5

sigmoid_thresholds=(0.001 0.01 0.02 0.05 0.1 0.25 0.5)
for sigmoid_threshold in ${sigmoid_thresholds[@]}
do
python -W ignore evaluate_voc_with_classifier.py --save-dir classifier/${sigmoid_threshold}  --restore-from snapshots/exp3/VOC_0_20000.pth --sigmoid-threshold ${sigmoid_threshold}

done




