CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --window_size 3 3 3 --is_training 1 --root_path ./dataset/pump/ --data_path sensor15m.csv --model_id nips --model $2 --data pump --features M --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 35 --dec_in 35 --c_out 35 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --window_size 4 4 4 --is_training 1 --root_path ./dataset/pump/ --data_path sensor15m.csv --model_id nips --model $2 --data pump --features M --seq_len 96 --label_len 24 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 35 --dec_in 35 --c_out 35 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --window_size 4 4 4 --is_training 1 --root_path ./dataset/pump/ --data_path sensor15m.csv --model_id nips --model $2 --data pump --features M --seq_len 336 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 35 --dec_in 35 --c_out 35 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --window_size 5 5 5 --is_training 1 --root_path ./dataset/pump/ --data_path sensor15m.csv --model_id nips --model $2 --data pump --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 35 --dec_in 35 --c_out 35 --des 'Exp' --itr 5
CUDA_VISIBLE_DEVICES=$1 python run.py --reweight $3 --window_size 5 5 5 --is_training 1 --root_path ./dataset/pump/ --data_path sensor15m.csv --model_id nips --model $2 --data pump --features M --seq_len 336 --label_len 168 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 35 --dec_in 35 --c_out 35 --des 'Exp' --itr 5