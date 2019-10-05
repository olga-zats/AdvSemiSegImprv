seed=0
while [ $seed -le 0 ]
do
    python train.py --snapshot-dir snapshots/exp0 --partial-data 0.125 --num-steps 20000 --lambda-adv-pred 0.0 --lambda-semi 0.0 --lambda-semi-adv 0.0 --mask-T 0.2 --random-seed ${seed}
    python evaluate_voc.py --restore-from snapshots/exp0/VOC_${seed}_20000.pth --save-dir exp0 --save-name results_${seed}.txt
    ((seed++))
done
