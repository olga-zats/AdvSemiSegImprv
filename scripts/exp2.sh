export CUDA_VISIBLE_DEVICES=0  
cd ..
seed=0
while [ $seed -le 0 ]
do
    python train.py --snapshot-dir snapshots/exp2 --partial-data 0.125 --num-steps 20000 --lambda-semi 0.0 --random-seed ${seed}
    python evaluate_voc.py --restore-from snapshots/exp2/VOC_${seed}_20000.pth --save-dir exp2 --save-name results_${seed}.txt
    ((seed++))
done
