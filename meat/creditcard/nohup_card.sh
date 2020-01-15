
if [ -z $1 ];then
    echo "Result file name is not assigned!"
    exit -1

echo "Result will be redirect to $1."

nohup python -u ./card.py --train_data=../../testdata/creditcard_fraud/creditcard_fraud_train.csv --test_data=../../testdata/creditcard_fraud/creditcard_fraud_eval.csv > $1 2>&1 & 