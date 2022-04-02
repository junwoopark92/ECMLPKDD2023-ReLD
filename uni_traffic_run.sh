sh uni_traffic.sh 2 Autoformer base > base_traffic_autoformer.log &
sh uni_traffic.sh 2 Informer base > base_traffic_informer.log &
sh uni_traffic.sh 0 Transformer base > base_traffic_transformer.log &
sh uni_traffic.sh 3 Autoformer lds > base_traffic_autoformer_lds.log &
sh uni_traffic.sh 3 Informer lds > base_traffic_informer_lds.log &
sh uni_traffic.sh 0 Transformer lds > base_traffic_transformer_lds.log &
