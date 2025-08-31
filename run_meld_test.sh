python -u main.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 \
    --batch-size 16 --graph_type='hyper' --epochs=0 --graph_construct='direct' \
    --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' \
    --norm BN --num_L=3 --num_K=3 --testing