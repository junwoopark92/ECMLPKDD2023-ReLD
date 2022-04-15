sh ano_traffic.sh 2 Autoformer base $1 > $1_traffic_autoformer.log &
sh ano_traffic.sh 2 Informer base $1 > $1_traffic_informer.log &
sh ano_traffic.sh 0 Transformer base $1 > $1_traffic_transformer.log &
sh ano_traffic.sh 3 Autoformer lds $1 > $1_traffic_autoformer_lds.log &
sh ano_traffic.sh 3 Informer lds $1 > $1_traffic_informer_lds.log &
sh ano_traffic.sh 0 Transformer lds $1 > $1_traffic_transformer_lds.log &
