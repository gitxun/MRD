python -u kdmain.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=5 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
      --modals='avl' --Dataset='MELD' --norm BN --num_L=3 --num_K=3 --seed 67137 \
      --calibrate --rank_coff 0.002 \
      --contrastlearning --mscl_coff 0.15 --cscl_coff 0.15 \
      --courselearning --epoch_ratio 0.4 --scheduler_steps 1