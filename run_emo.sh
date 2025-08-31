python kdmain.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type 'hyper' \
      --epochs 10 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT'\
      --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L 3 --num_K 4 --seed 1475 \
      --calibrate --rank_coff 0.005 \
      --contrastlearning --mscl_coff 0.05 --cscl_coff 0.05 \
      --courselearning --epoch_ratio 0.15 --scheduler_steps 1