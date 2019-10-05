export CUDA_VISIBLE_DEVICES=0
seeds=(0)
for seed in ${seeds[*]}
do
   python gl_pyramid_train.py --snapshot-dir snapshots/gl_pyramid --partial-data 0.125 --num-steps 20000  --random-seed $seed  --gpu 0
   python evaluate_voc.py --restore-from snapshots/gl_pyramid/VOC_20000_${seed}.pth --save-dir gl_pyramid --save-name results_${seed}.txt --gpu 0  
done
