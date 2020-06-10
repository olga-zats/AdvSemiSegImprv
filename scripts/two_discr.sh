export CUDA_VISIBLE_DEVICES=0   
seed=0
cd ..
while [ $seed -le 0 ]
do
    python two_discr_train.py --snapshot-dir snapshots/two_discr --partial-data 0.125 --num-steps 20000 --discr-split 5000 --batch-size 5 --random-seed $seed --lambda-semi 0.0
    python evaluate_voc.py --restore-from snapshots/two_discr/VOC_20000_${seed}.pth --save-dir two_discr --save-name results_${seed}.txt
    ((seed++))
done
