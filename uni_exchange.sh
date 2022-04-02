CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id gap3 --model $2 --data exchange --features S --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id gap3 --model $2 --data exchange --features S --seq_len 48 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id gap3 --model $2 --data exchange --features S --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model_id gap3 --model $2 --data exchange --features S --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5