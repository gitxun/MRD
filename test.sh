python -u model_vs.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size 16 --graph_type='hyper' \
      --epochs=0 --graph_construct='direct' --multi_modal \
      --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --testing