# Reproducing the ReLD experiment results
This is an official repository for the paper, titled [Deep Imbalanced Time-series Forecasting via Local Discrepancy Density](https://arxiv.org/abs/2302.13563), accepted at ECML/PKDD'23
## Environment setup
```
python == 3.7
pip install -r requirements.txt
```

## ETT dataset with univariate setting
```
# Original
python run.py --window_size 3 3 3 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5

# ReLD 
python run.py --window_size 3 3 3 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id test --model Autoformer --data ETTm1 --features S --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 1 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 5

```

## ETT dataset with multivariate setting

```
# Original
python run.py --window_size 3 3 3 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5

# ReLD 
python run.py --window_size 3 3 3 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id test --model Autoformer --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5

```

## Other datasets with multivariate setting

```
datasetdir = {pump, airquality, weather, commodity}
datasetname = {sensor15m, airquality, weather_h, Corn}  

# Original
python run.py --window_size 3 3 3 --reweight base --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight base --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight base --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5

# ReLD 
python run.py --window_size 3 3 3 --reweight lds --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 48 --label_len 24 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 4 4 4 --reweight lds --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5
python run.py --window_size 5 5 5 --reweight lds --is_training 1 --root_path ./dataset/{datasetdir}/ --data_path {datasetname}.csv --model_id test --model Autoformer --data custom --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 5

```

## Citation
```
@inproceedings{park23deep,
  publisher = {Springer Nature Switzerland},
  booktitle = {Proc. of Machine Learning and Knowledge Discovery in Databases: Research Track (ECML/PKDD)},
  author    = {Junwoo Park and Jungsoo Lee and Youngin Cho and Woncheol Shin and Dongmin Kim and Jaegul Choo and Edward Choi},
  title     = {Deep Imbalanced Time-Series Forecasting via Local Discrepancy Density},
  year      = {2023},
  doi       = {10.1007/978-3-031-43424-2_9},
  pages     = {139-155},
  url       = {https://link.springer.com/content/pdf/10.1007/978-3-031-43424-2_9}
}
```
 