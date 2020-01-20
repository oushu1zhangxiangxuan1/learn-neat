from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

import numpy as np


def test():
    le = LabelEncoder()

    res = le.fit_transform(['a','b','c'])

    print(res)


    enc = OneHotEncoder()

    # enc.fit()

    res = enc.fit_transform([[0],[1],[2]])

    print(res.toarray())


class ZxxEncoder():
    def __init__(self, labels=None):
        assert labels is not None
        self.le = LabelEncoder()
        self.le.fit(labels)
        res = self.le.transform(labels)
        self.oh = OneHotEncoder()
        self.oh.fit(np.reshape(res,(-1,1)))
        self.tmp_out = self.oh.transform(np.reshape(res,(-1,1)))

    def transform(self, y):
        tmp = self.le.transform(y)
        print(tmp)
        res = self.oh.transform(np.reshape(tmp, (-1,1))).toarray()
        print(res)
        return res

    def inverse_transform_v1(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res

    def inverse_transform(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res
    
def TestZxxEncoder():
    labels = ['a','b','c']

    candis = ['c','a','a','b']

    ze = ZxxEncoder(labels)

    res = ze.transform(candis)

    print(res)

    candis = ['c'] 
    res = ze.transform(candis)

    print(res)

    # inverse_candis = [
    #     [0., 0., 1.],
    #     [1., 0., 0.],
    #     [1., 0., 0.],
    #     [0., 1., 0.]
    # ]

    # res = ze.inverse_transform(inverse_candis)

    # print(res)

def TestNumpyReshape():

    a = [2,3,4]

    print(np.shape(a))

    print(np.reshape(a, (-1,1)))


    a = [2]

    print(np.shape(a))

    print(np.reshape(a, (-1,1)))


def printObjectDict(obj):
    for k,v in obj.__dict__.items():
        print(k)
        print(v)
        print()

def traverseOneHotEncoder():
    ohe = OneHotEncoder()
    printObjectDict(ohe)
    print(ohe.fit_transform([
        [1],[5],[9]
    ]).toarray())
    print(ohe.fit_transform([
        [1],[5],[9]
    ]))
    printObjectDict(ohe)



    print("\n\n-----------------------\n\n")
    X = [
        [1],[5],[9]
    ]
    out = ohe.fit_transform(X)
    printObjectDict(out)

    print(out.sorted_indices())

    decode_columns = np.vectorize(lambda col: ohe.active_features_[col])
    print("decode_columns:")
    print(decode_columns)

    decoded = decode_columns(out.indices).reshape(np.shape(X))
    print("decoded")
    print(decoded)

    recovered_X = decoded - ohe.feature_indices_[:-1]
    print("recovered_X")
    print(recovered_X)

class ZxxEncoder_v2():
    def __init__(self, labels):
        if DEBUG:
            print(labels)
        assert labels is not None
        self.le = LabelEncoder()
        self.le.fit(labels)
        res = self.le.transform(labels)
        self.oh = OneHotEncoder()
        self.oh.fit(np.reshape(res,(-1,1)))
        self.tmp_out = self.oh.transform(np.reshape(res,(-1,1)))

    def transform(self, y):
        tmp = self.le.transform(y)
        if DEBUG:
            print(tmp)
        res = self.oh.transform(np.reshape(tmp, (-1,1))).toarray()
        if DEBUG:
            print(res)
        return res

    def inverse_transform_v1(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res

    def inverse_transform(self, y):

        decode_columns = np.vectorize(lambda col: self.oh.active_features_[col])
        print("decode_columns:")
        print(decode_columns)


        print(np.shape(y))
        print(np.shape(self.tmp_out))

        decoded = decode_columns(y.indices).reshape(-1,np.shape(self.tmp_out)[-1])
        print("decoded")
        print(decoded)

        recovered_y = decoded - self.oh.feature_indices_[:-1]

        res = self.le.inverse_transform(recovered_y)
        print(res)
        return res
 



if "__main__" == __name__:

    # TestNumpyReshape()

    TestZxxEncoder()

    # traverseOneHotEncoder()

