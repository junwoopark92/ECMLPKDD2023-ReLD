CUDA_VISIBLE_DEVICES=$1 python run.py  --denorm --reweight $4 --is_training 1 --root_path ./dataset/ETT-small/ --data_path $3.csv --model_id $3 --model $2 --data custom --features S --seq_len 192 --label_len 96 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py  --denorm --reweight $4 --is_training 1 --root_path ./dataset/ETT-small/ --data_path $3.csv --model_id $3 --model $2 --data custom --features S --seq_len 192 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py  --denorm --reweight $4 --is_training 1 --root_path ./dataset/ETT-small/ --data_path $3.csv --model_id $3 --model $2 --data custom --features S --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py  --denorm --reweight $4 --is_training 1 --root_path ./dataset/ETT-small/ --data_path $3.csv --model_id $3 --model $2 --data custom --features S --seq_len 336 --label_len 96 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5