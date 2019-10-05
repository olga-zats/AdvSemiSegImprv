export CUDA_VISIBLE_DEVICES=0   
sizes=(107  321)
for size in ${sizes[*]}
do
   python ave_pool_train.py --snapshot-dir snapshots/ave_pool --partial-data 0.125 --num-steps 20000  --random-seed 0 --kernel-size ${size},${size} --gpu 0
   python evaluate_voc.py --restore-from snapshots/ave_pool/VOC_20000_${size}_0.pth --save-dir ave_pool --save-name results_${size}_0.txt --gpu 0
done
