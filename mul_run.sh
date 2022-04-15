sh mul.sh 0 $1 Transformer lds > mul_$1_transformer_lds.log &
sh mul.sh 2 $1 Informer lds > mul_$1_Informer_lds.log &
sh mul.sh 2 $1 Autoformer lds > mul_$1_autoformer_lds.log &
sh mul.sh 0 $1 Transformer base > mul_$1_transformer.log &
sh mul.sh 3 $1 Informer base > mul_$1_Informer.log &
sh mul.sh 3 $1 Autoformer base > mul_$1_autoformer.log &

