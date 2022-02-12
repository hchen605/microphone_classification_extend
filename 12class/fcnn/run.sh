for i in {1..10..1}
do
    python train_fcnn.py --gender 0 --nclass 1 --limit 5 --eps 200 
done
