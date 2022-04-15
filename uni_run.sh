sh uni.sh 0 $1 Transformer lds > $1_transformer_lds.log &
sh uni.sh 1 $1 Informer lds > $1_Informer_lds.log &
sh uni.sh 5 $1 Autoformer lds > $1_autoformer_lds.log &
sh uni.sh 0 $1 Transformer base > $1_transformer.log &
sh uni.sh 5 $1 Informer base > $1_Informer.log &
sh uni.sh 1 $1 Autoformer base > $1_autoformer.log &

