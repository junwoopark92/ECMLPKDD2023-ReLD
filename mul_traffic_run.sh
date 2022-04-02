sh mul_traffic.sh 2 Autoformer base > mul_traffic_autoformer.log &
sh mul_traffic.sh 3 Informer base > mul_traffic_informer.log &
sh mul_traffic.sh 1 Transformer base > mul_traffic_transformer.log &
sh mul_traffic.sh 3 Autoformer lds > mul_traffic_autoformer_lds.log &
sh mul_traffic.sh 2 Informer lds > mul_traffic_informer_lds.log &
sh mul_traffic.sh 0 Transformer lds > mul_traffic_transformer_lds.log &
