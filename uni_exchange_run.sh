sh uni_exchange.sh 1 Autoformer base > exchange_autoformer.log &
sh uni_exchange.sh 2 Informer base > exchange_informer.log &
sh uni_exchange.sh 3 Transformer base > exchange_transformer.log &
sh uni_exchange.sh 1 Autoformer lds > exchange_autoformer_lds.log &
sh uni_exchange.sh 3 Informer lds > exchange_informer_lds.log &
sh uni_exchange.sh 2 Transformer lds > exchange_transformer_lds.log &
