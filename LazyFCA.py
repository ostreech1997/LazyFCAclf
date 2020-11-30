# Packages 
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def preprocessing(file_name='Data/tic-tac-toe-endgame.csv', target_column = 'V10',
                  train_part=0.8, shuffle = True, class_labels=['negative','positive']):
    # Load data from file
    data = pd.read_csv(file_name)
    # Random shuffle
    if shuffle:
        data = data.sample(frac=1)
        
    data = data.reset_index(drop=True)
    m, n = data.shape
    # Split data
    for col in data.drop(target_column, axis=1):
        data.loc[data[col]!='x',col] = 0
        data.loc[data[col]=='x',col] = 1
        
    train = data.iloc[:int(train_part*m),:]
    test = data.iloc[int(train_part*m):,:]
    
    train = train.replace(class_labels[1], 1)
    train = train.replace(class_labels[0], 0)

    test = test.replace(class_labels[1], 1)
    test = test.replace(class_labels[0], 0)    
    
    X_train = train.drop(['V10'], axis=1)
    Y = train['V10']
    
    X_test = test.drop(['V10'], axis=1)
    
    Y_test = test['V10'].to_numpy()
    Y_test = Y_test.astype('int64')
    
    
    # Devide train sample into C_plus and C_minus
    
    C_test = X_test.to_numpy()
    
    C_plus = X_train[Y == 1].to_numpy()
    C_minus = X_train[Y == 0].to_numpy()
    
    data_dict = {'C_plus':C_plus, 'C_minus':C_minus, 'C_test':C_test, 'Y_test':Y_test}
    
    return data_dict
    

# check 
def intersection_func(a1,a2):
    output = np.equal(a1,a2)
    output_ind = [i for i,x in enumerate(output) if x]
    return output, output_ind

def calculate_intersection(C, example, intersection_ind, max_int = 1):
    k = 0
    for x in C:
        intersection_i, intersection_ind_i = intersection_func(example, x)
#         print('BIG:',intersection_ind)
#         print('SMALL:',intersection_ind_i)
        if set(intersection_ind).issubset(set(intersection_ind_i)):
            k += 1
            if k >= max_int:
                return 0
    return 1

def LazyFCAclf(C_plus,C_minus,new_example, max_int = 1):
    num_pos = 0
    num_neg = 0

    for x in C_plus:
        intersection, intersection_ind = intersection_func(new_example, x)
        num_pos += calculate_intersection(C_minus, new_example, intersection_ind, max_int)

    for x in C_minus:
        intersection, intersection_ind = intersection_func(new_example, x)
        num_neg += calculate_intersection(C_plus, new_example, intersection_ind, max_int)
    
    if num_pos >= num_neg:        
        return 1   
    else:        
        return 0    


def Predict(data_dict, max_int):
    
    Y_pred = []
    for x in tqdm(data_dict['C_test']):
        Y_pred.append(LazyFCAclf(data_dict['C_plus'], data_dict['C_minus'], x, max_int))
    
    Y_pred = np.array(Y_pred)
    
#     print('ACCURACY:', accuracy_score(y_true=data_dict['Y_test'], y_pred=Y_pred))    
#     print('PRECISION:', precision_score(y_true=data_dict['Y_test'], y_pred=Y_pred))
#     print('RECALL:', recall_score(y_true=data_dict['Y_test'], y_pred=Y_pred))
#     print('ROC_AUC:', roc_auc_score(y_true=data_dict['Y_test'], y_score=Y_pred))

    return Y_pred

def Metrics(Y_true, Y_pred):
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['precision'] = precision_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['recall'] = recall_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['ROC_AUC'] = roc_auc_score(y_true=Y_true, y_score=Y_pred)
  
    return metrics_dict
    