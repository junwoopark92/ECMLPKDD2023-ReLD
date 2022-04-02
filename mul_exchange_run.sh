sh mul_exchange.sh 1 Autoformer base > mul_exchange_autoformer.log &
sh mul_exchange.sh 2 Informer base > mul_exchange_informer.log &
sh mul_exchange.sh 3 Transformer base > mul_exchange_transformer.log &
sh mul_exchange.sh 2 Autoformer lds > mul_exchange_autoformer_lds.log &
sh mul_exchange.sh 1 Informer lds > mul_exchange_informer_lds.log &
sh mul_exchange.sh 3 Transformer lds > mul_exchange_transformer_lds.log &
