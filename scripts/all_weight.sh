  
#!/bin/bash
for DIM in 512
do
    for data in cub car
    do
        for rot in 0.05 0.1 0.5
        do
            for rot_batch in 16
            do
                for rot_lr in 0.000005 0.00001
                do
                    for num_clusters in 100
                    do
                    sbatch -o ./results_center_crop_cluster/rot/%j_${DIM}_${data}_${rot}_${rot_lr}_${rot_batch}_${num_clusters} run_train_00.sh $rot $data Weight ${rot_lr} 40 1 ${rot_batch} 0.00001 $DIM 5 ${num_clusters}
                    #./run_train_00.sh 0.01 cub Weight 0.00001 40 1 16 0.00001 512 5 100
                    done
                done
            done 
        done
    done
done