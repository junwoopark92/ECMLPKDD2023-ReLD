sh mul.sh $1 $5 Pyraformer onlyld > mul_$5_Pyraformer_onlyld.log &
sh mul.sh $2 $5 Autoformer onlyld > mul_$5_Autoformer_onlyld.log &
sh mul.sh $3 $5 SCINet onlyld > mul_$5_SCINet_onlyld.log &
sh mul.sh $4 $5 Informer onlyld > mul_$5_Informer_onlyld.log &