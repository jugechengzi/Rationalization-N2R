## Environments
torch 1.13.1+cu11.6.  
python 3.7.16.   

## Datasets
Beer: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running example  
For N2R on Beer-Appearance:  
python -u mochang_noacc.py--seed $seed --add 0 --dis_lr 0 --hidden_dim 200 --data_type beer --lr 0.0001 --batch_size 128 --sparsity_percentage 0.125 --sparsity_lambda 11 --continuity_lambda 12 --mochang_lambda 1 --mochang_loss log --epochs 300 --aspect 0 

For N2R+RNP:  
python -u mochang.py --seed $seed --add 0 --dis_lr 0 --hidden_dim 200 --data_type beer --lr 0.0001 --batch_size 128 --sparsity_percentage 0.125 --sparsity_lambda 11 --continuity_lambda 12 --mochang_lambda 1 --mochang_loss vanilla --epochs 300 --aspect 0


Notes: You need to replace $seed with [1,2,3,4,5] to get results with different random seeds. If you want to take our paper as a baseline, we recommend using N2R+RNP, as it usually performs better than N2R alone.