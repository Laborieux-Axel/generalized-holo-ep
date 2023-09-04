

# Table 1
for seed in 0 1 2 3 4
do
    for beta in 0.05 0.5
    do
        for N in 1 2 4 6
        do
            python train.py \
                --discrete --seed $seed \
                --rotate --angle 0.0 \
                --net mlp --arch mlp_2-256 --algo hep_vf --N $N --dataset fmnist \
                --bsz 50 --act sigmoid --opt sgd --loss xent --mmt 0.9 \
                --lr 0.01 --wd 0.0 --beta $beta --T1 150 --T2 20 --num_epochs 50

        done
    done 

	python train.py \
		--discrete --name mlp_fmnist_online-hvf --seed $seed \
		--net mlp --arch mlp_2-256 --algo online_hvf --N 4 --dataset fmnist \
		--rotate --angle 0.0 \
		--bsz 50 --act sigmoid --opt sgd --loss xent --mmt 0.9 \
		--lr 0.01 --wd 0.0 --beta $beta --T1 1200 --T2 60 --num_epochs 50
done


# Fig 3
for seed in 0 1 2
do
    for angle in 0.0 15.0 30.0 45.0 60.0 75.0 90.0 
    do
        python train.py \
            --discrete --seed $seed \
            --rotate --angle $angle \
            --net mlp --arch mlp_2-256 --algo bp --N 0 --dataset fmnist \
            --bsz 50 --act sigmoid --opt sgd --loss xent --mmt 0.9 \
            --lr 0.01 --wd 0.0 --T1 150 --T2 20 --num_epochs 50

        python train.py \
            --discrete --seed $seed \
            --rotate --angle $angle \
            --net mlp --arch mlp_2-256 --algo hep_vf --N 0 --dataset fmnist \
            --bsz 50 --act sigmoid --opt sgd --loss xent --mmt 0.9 \
            --lr 0.01 --wd 0.0 --T1 150 --T2 20 --num_epochs 50
    done
done

# fig 4 a-d)
for seed in 0 1 2 3 4
do
    for jr in 0.0 1.0
    do
        python train.py \
            --discrete --seed $seed \
            --rotate --angle 90 \
            --jac_reg --jac_reg_coef $jr \
            --net mlp --arch mlp_2-256 --algo hep_vf --N 0 --dataset fmnist \
            --bsz 50 --act sigmoid --opt adamw --loss xent \
            --lr 1e-4 --wd 0.0 --T1 150 --T2 20 --num_epochs 50
    done
done
 
 
# Fig 4 e-h)
for seed in 0 1 2 3 4
do
    for jr in 0.0 0.1
    do
        python train.py \
            --discrete --seed $seed \
            --jac_reg --jac_reg_coef $jr \
            --net mlp --arch loop_mnist --algo hep_vf --N 0 --dataset fmnist \
            --bsz 50 --act sigmoid --opt sgd --loss xent --mmt 0.9 \
            --lr 1e-2 --wd 0.0 --T1 150 --T2 20 --num_epochs 50
    done
done


# Fig 4 i-l)
for seed in 1 2 3
do
    python train.py \
        --algo hep_vf --half_prec --seed $seed --parallel           \
        --dataset cifar100 --net cnn --act dsilu                    \
        --rotate --angle 0.0                                        \
        --opt sgd_cos --bsz 256 --lr 5e-3 --wd 1e-4  --mmt 0.9      \
        --loss xent --beta 1.0 --T1 250 --T2 40 --N 0               \
        --num_epochs 90

    python train.py \
        --algo hep_vf --half_prec --seed $seed --parallel           \
        --dataset cifar10 --net cnn --act dsilu                     \
        --rotate --angle 0.0                                        \
        --jac_reg --jac_reg_coef 15.0                               \
        --opt sgd_cos --bsz 256 --lr 5e-3 --wd 1e-4  --mmt 0.9      \
        --loss xent --beta 1.0 --T1 250 --T2 40 --N 0               \
        --num_epochs 90
done


# Table 2


for seed in 0 1 2
do
    for dataset in cifar10 cifar100
    do
        # hEP with ground truth dudbeta
        python train.py \
            --algo hep_vf --half_prec --seed $seed --parallel           \
            --dataset $dataset --net cnn --act dsilu                    \
            --rotate --angle 0.0                                        \
            --jac_reg --jac_reg_coef 15.0                               \
            --opt sgd_cos --bsz 256 --lr 7e-3 --wd 1e-4  --mmt 0.9      \
            --loss xent --T1 250 --T2 40 --N 0                          \
            --num_epochs 150 &

		# hEP N = 2
        python train.py \
            --algo hep_vf --half_prec --seed $seed --parallel           \
            --dataset $dataset --net cnn --act dsilu                    \
            --rotate --angle 0.0                                        \
            --jac_reg --jac_reg_coef 15.0                               \
            --opt sgd_cos --bsz 256 --lr 7e-3 --wd 1e-4  --mmt 0.9      \
            --loss xent --beta 1.0 --T1 250 --T2 40 --N 2               \
            --num_epochs 150 &
    
		# BP with homeo
        python train.py \
            --algo bp --half_prec --seed $seed --parallel               \
            --dataset $dataset --net cnn --act dsilu                    \
            --rotate --angle 0.0                                        \
            --jac_reg --jac_reg_coef 15.0                               \
            --opt sgd_cos --bsz 256 --lr 7e-3 --wd 1e-4  --mmt 0.9      \
            --loss xent --T1 250 --T2 40 --N 0                          \
            --num_epochs 150 &
    
		# BP without homeo
        python train.py \
            --algo bp --half_prec --seed $seed --parallel               \
            --dataset $dataset --net cnn --act dsilu                    \
            --rotate --angle 0.0                                        \
            --opt sgd_cos --bsz 256 --lr 7e-3 --wd 1e-4  --mmt 0.9      \
            --loss xent --T1 250 --T2 40 --N 0                          \
            --num_epochs 150 &
    done
done




for seed in 0 1 2
do
    python train.py \
        --parallel --algo hep_vf --half_prec --seed $seed       \
        --dataset imagenet32 --net cnn --act dsilu              \
        --rotate --angle 0.0                                    \
        --jac_reg --jac_reg_coef 10.0                           \
        --opt sgd_cos --bsz 256 --lr 5e-3 --wd 1e-4 --mmt 0.9   \
        --loss xent --T1 250 --T2 40 --N 0                      \
        --num_epochs 90
done



