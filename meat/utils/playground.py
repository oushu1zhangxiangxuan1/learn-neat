
def Test_OnehotEncoder_Inverse_Transform():
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    print(enc.categories_)
    enc.transform([['Female', 1], ['Male', 4]]).toarray()
    enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    enc.get_feature_names(['gender', 'group'])
    drop_enc = OneHotEncoder(drop='first').fit(X)
    drop_enc.categories_
    drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()



if __name__ == '__main__':
    Test_OnehotEncoder_Inverse_Transform()