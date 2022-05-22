sh uni_ano.sh $1 $5 Pyraformer $6 > uni_$5_Pyraformer_$6.log &
sh uni_ano.sh $2 $5 Autoformer $6 > uni_$5_Autoformer_$6.log &
sh uni_ano.sh $3 $5 Nbeats $6 > uni_$5_Nbeats_$6.log &
sh uni_ano.sh $4 $5 Informer $6 > uni_$5_Informer_$6.log &