for OL in {1..30}
do
  for IL in {1..30}
    do
      CUDA_VISIBLE_DEVICES=$1 python run.py --reweight base --window_size 4 4 4 --is_training 1 --root_path ./dataset/ETT-small/ --data_path $2.csv --model_id iovar --model $3  --data $2  --features S --seq_len $((IL * 16)) --label_len $((IL * 8)) --pred_len $((OL * 16)) --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1  --c_out 1  --des 'Exp'  --itr 3
    done
done