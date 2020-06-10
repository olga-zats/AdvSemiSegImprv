export CUDA_VISIBLE_DEVICES=0   
cd ..
alphas=(0.51 0.7 0.9)
for alpha in ${alphas[*]}
do
    python smooth_gt_train.py --snapshot-dir snapshots/smooth_gt --partial-data 0.125 --num-steps 20000 --random-seed 0 --interp-alpha $alpha  --lambda-semi 0.0 --gpu 0
    python evaluate_voc.py --restore-from snapshots/smooth_gt/VOC_20000_0_${alpha}.pth --save-dir smooth_gt --save-name results_0_{$alpha}.txt --gpu 0
done
