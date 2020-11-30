from LazyFCA import preprocessing, Predict

if __name__ == '__main__':

    data_dict = preprocessing()
    y_pred = Predict(data_dict)