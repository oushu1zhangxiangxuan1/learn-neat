# echo "result will be write to $1"
# nohup python -u ./iris.py --train_data=../../testdata/iris_feature_selection/iris_train.csv --test_data=../../testdata/iris_feature_selection/iris_test.csv > $1 2>&1 & 

nohup python -u ./iris.py --train_data=../../testdata/iris_feature_selection/ones/iris_train.csv --test_data=../../testdata/iris_feature_selection/ones/iris_test.csv --name=ones > ones.out 2>&1 & 
nohup python -u ./iris.py --train_data=../../testdata/iris_feature_selection/random/iris_train.csv --test_data=../../testdata/iris_feature_selection/random/iris_test.csv --name=random > random.out 2>&1 & 