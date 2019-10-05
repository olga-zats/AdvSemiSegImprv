export CUDA_VISIBLE_DEVICES=0   
seed=0
while [ $seed -le 0 ]
do
    python train.py --snapshot-dir snapshots/exp3 --partial-data 0.125 --num-steps 20000 --lambda-adv-pred 0.01 --lambda-semi 0.1 --lambda-semi-adv 0.001 --mask-T 0.2 --random-seed ${seed} 
    python evaluate_voc.py --restore-from snapshots/exp3/VOC_${seed}_20000.pth --save-dir exp3 --save-name results_${seed}.txt 
    ((seed++))
done
