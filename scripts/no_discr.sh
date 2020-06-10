export CUDA_VISIBLE_DEVICES=0   
cd ..
for lambda in $(seq 0.2 0.8 0.2)
do
    seed=0
    while [ $seed -le 0 ]
    do
        python no_discr_train.py --snapshot-dir snapshots/no_discr --partial-data 0.125 --num-steps 20000 --lambda-semi ${lambda} --random-seed ${seed} --semi-start 5000 
        python evaluate_voc.py --restore-from snapshots/no_discr/VOC_20000_${lambda}_${seed}.pth --save-dir no_discr --save-name results_${lambda}_${seed}.txt 
        ((seed++))
    done
done
