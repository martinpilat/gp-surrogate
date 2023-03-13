 
 
 

for prob_num in [7,8,9,10]:
    for ts in [1000, 5000]:
        for re in [1,3,5]:
            print(f'qsub -l walltime={48 if ts == 5000 else 24}:0:0 -l select=1:ncpus=16:mem=8gb -v', end='')
            print(f"\"PROB_NUM={prob_num},SURROGATE_NAME=TNN,ADDITIONAL_OPTIONS='--filename TS{ts}_RE{re} --retrain_every {re} -N 20 --max_train_size {ts}'\" run_surrogate.sh")