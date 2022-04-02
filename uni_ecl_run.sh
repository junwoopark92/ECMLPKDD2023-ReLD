sh uni_ecl.sh 1 Autoformer base > base_ecl_autoformer.log &
sh uni_ecl.sh 2 Informer base > base_ecl_informer.log &
sh uni_ecl.sh 3 Transformer base > base_ecl_transformer.log &
sh uni_ecl.sh 3 Autoformer lds > base_ecl_autoformer_lds.log &
sh uni_ecl.sh 2 Informer lds > base_ecl_informer_lds.log &
sh uni_ecl.sh 1 Transformer lds > base_ecl_transformer_lds.log &
