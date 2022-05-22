sh uni.sh $1 $5 Pyraformer onlyld > uni_$5_Pyraformer_onlyld.log &
sh uni.sh $2 $5 Autoformer onlyld > uni_$5_Autoformer_onlyld.log &
sh uni.sh $3 $5 SCINet onlyld > uni_$5_SCINet_onlyld.log &
sh uni.sh $4 $5 Informer onlyld > uni_$5_Informer_onlyld.log &