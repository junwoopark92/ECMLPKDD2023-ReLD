sh ano_toy.sh 0 Autoformer $1 base > $1_autoformer.log &
sh ano_toy.sh 2 Informer $1 base > $1_informer.log &
sh ano_toy.sh 3 Transformer $1 base > $1_transformer.log &
sh ano_toy.sh 0 Autoformer $1 lds > $1_autoformer_lds.log &
sh ano_toy.sh 2 Informer $1 lds > $1_informer_lds.log &
sh ano_toy.sh 3 Transformer $1 lds > $1_transformer_lds.log &
