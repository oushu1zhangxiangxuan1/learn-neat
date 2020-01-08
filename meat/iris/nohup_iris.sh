echo "result will be write to $1"

nohup python -u ./iris.py --train_data=testdata/iris_train.csv --test_data=testdata/iris_test.csv > $1 2>&1 & 