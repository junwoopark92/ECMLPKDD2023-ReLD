python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_Pyraformer_672_720 \
  --model Pyraformer \
  --data ETTm1 \
  --features M \
  --seq_len 672 \
  --label_len 0 \
  --pred_len 672 \
  --CSCM Bottleneck_Construct \
  --pyraformer_decoder_type FC \
  --pyraformer_embed_type DataEmbedding \
  --e_layers 4 \
  --enc_in 7 \
  --covariate_size 4 \
  --d_bottleneck 64 \
  --d_model 256 \
  --d_inner_hid 512 \
  --d_k 64 \
  --d_v 64 \
  --dropout 0.2 \
  --n_heads 6 \
  --seq_num 1 \
  --inner_size 3 \
  --window_size 6 6 6 \
  --itr 1
  # --batch_size 16 \
  # --lr_decay_factor 0.1 \