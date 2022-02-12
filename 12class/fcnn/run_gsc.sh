# +
#for i in {1..10..1}
#do
#    python train_fcnn_gsc.py --gender 0 --nclass 0 --limit 5 --eps 200 
#done


#for i in {1..10..1}
#do
#    python train_fcnn_gsc.py --gender 0 --nclass 0 --limit 25 --eps 200 
#done

#for i in {1..10..1}
#do
#    python train_fcnn_gsc.py --gender 0 --nclass 0 --limit 50 --eps 200 
#done

for i in {5..10..1}
do
    python train_fcnn_gsc.py --gender 0 --nclass 1 --limit 5 --eps 200 
done


for i in {1..10..1}
do
    python train_fcnn_gsc.py --gender 0 --nclass 1 --limit 25 --eps 200 
done

for i in {1..10..1}
do
    python train_fcnn_gsc.py --gender 0 --nclass 1 --limit 50 --eps 200 
done



