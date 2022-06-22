for OL in {1..25}
do
  for IL in {1..25}
    do
      CUDA_VISIBLE_DEVICES=$1 python run.py --reweight base --window_size 3 3 3 --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id ivar --model $2 --data exchange --features S --seq_len $((IL * 16)) --label_len $((IL * 8)) --pred_len $((IL * 16)) --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 3
    done
done