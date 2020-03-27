  
#!/bin/bash
for DIM in 512
do
    for data in cub
    do
        for rot in 0.1
        do
            for rot_batch in 16
            do
                for rot_lr in 0.000005
                do
                    for num_clusters in 100
                    do
                    sbatch -o verify.out run_train_test.sh $rot $data Weight ${rot_lr} 40 1 ${rot_batch} 0.00001 $DIM 5 ${num_clusters}
                    done
                done
            done 
        done
    done
done