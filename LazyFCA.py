# Packages 
import pandas as pd
import numpy as np
import math 
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

# Categorical -> Binary 
def preprocessing(df, target_column, target_dict, shuffle = True):
    
    df2 = df.copy()
    if shuffle:
        df2 = df2.sample(frac=1)
        df2 = df2.reset_index(drop=True)
    for col in df2:
        unique_values = df2[col].unique()
        if len(unique_values) == 2:
            if col == target_column:
                df2[col] = df2[col].replace(list(target_dict.keys())[0],0)
                df2[col] = df2[col].replace(list(target_dict.keys())[1],1)
            else:
                df2[col] = df2[col].replace(unique_values[0],0)
                df2[col] = df2[col].replace(unique_values[1],1)
        else:
            OHE = OneHotEncoder()
            col_arr = OHE.fit_transform(df2[col].to_numpy().reshape(-1,1)).toarray()
            df2 = df2.drop([col], axis=1)
            names = [col + '_' + it for it in OHE.get_feature_names()]
            df_col = pd.DataFrame(col_arr, columns=names)
            df2 = df2.join(df_col)
    
    return df2
    

# check 
def intersection_func(a1,a2):
    output = np.equal(a1,a2)
    output_ind = [i for i,x in enumerate(output) if x]
    return output, output_ind

def calculate_intersection(C, example, intersection_ind, max_int = 1):
    k = 0
    for x in C:
        intersection_i, intersection_ind_i = intersection_func(example, x)
        
        if set(intersection_ind).issubset(set(intersection_ind_i)):
            k += 1
            if k >= max_int:
                return 0
    return 1

def LazyFCAclf(C_plus,C_minus,new_example, max_int = 1, min_elems = 0, balance = False, prop = 1):
    m = C_plus.shape[0]
    examples = random.sample(range(m),int(prop*m))
    C_plus = C_plus[examples]
    
    m = C_minus.shape[0]
    examples = random.sample(range(m),int(prop*m))
    C_minus = C_minus[examples]
    
    num_pos = 0
    num_neg = 0
    m_plus, n = C_plus.shape
    m_minus, n = C_minus.shape
    for x in C_plus:
        intersection, intersection_ind = intersection_func(new_example, x)
        if len(intersection_ind) < math.floor(n * min_elems):
            continue
        num_pos += calculate_intersection(C_minus, new_example, intersection_ind, max_int)

    for x in C_minus:
        intersection, intersection_ind = intersection_func(new_example, x)
        if len(intersection_ind) < math.floor(n * min_elems):
            continue
        num_neg += calculate_intersection(C_plus, new_example, intersection_ind, max_int)
    
    if balance:
        num_pos /= m_plus
        num_neg /= m_minus
    
    if num_pos >= num_neg:        
        return 1   
    else:        
        return 0    


def Predict(data_dict, max_int, min_elems=0, balance=False, prop=1):
    
    Y_pred = []
    for x in tqdm(data_dict['C_test']):
        Y_pred.append(LazyFCAclf(data_dict['C_plus'], data_dict['C_minus'], x, max_int, min_elems, balance, prop))
    
    Y_pred = np.array(Y_pred)
    

    return Y_pred

def Metrics(Y_true, Y_pred):
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['precision'] = precision_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['recall'] = recall_score(y_true=Y_true, y_pred=Y_pred)
    metrics_dict['ROC_AUC'] = roc_auc_score(y_true=Y_true, y_score=Y_pred)
  
    return metrics_dict


def cross_validation(df, target_column, Kfolds, shuffle, model=None, model_params={'max_int':1,
                                                                                   'min_elems':0,
                                                                                   'balance':False,
                                                                                   'prop':1}):
    
    accuracy = []
    precision = []
    recall = []
    ROC_AUC = []
    
    
    df2 = df.copy()
    
    if shuffle:
        df2 = df2.sample(frac=1)
        df2 = df2.reset_index(drop=True)
    
    m,n = df2.shape
    folds = [0] + [(m // Kfolds)*i for i in range(1,Kfolds)] + [m]
    
    for k in range(Kfolds):
        
        test_index = list(range(folds[k], folds[k+1]))

        data = df2.copy()

        train = data.drop(test_index, axis=0)
        test = data.iloc[test_index,:]

        X_train = train.drop([target_column], axis=1)
        Y = train[target_column]

        X_test = test.drop([target_column], axis=1)

        Y_test = test[target_column].to_numpy()
        Y_test = Y_test.astype('int64')


        if model == 'FCA':
        # Devide train sample into C_plus and C_minus        
            C_test = X_test.to_numpy()

            C_plus = X_train[Y == 1].to_numpy()
            C_minus = X_train[Y == 0].to_numpy()

            data_dict = {'C_plus':C_plus, 'C_minus':C_minus, 'C_test':C_test, 'Y_test':Y_test}
            Y_pred = Predict(data_dict,max_int=model_params['max_int'],
                             min_elems=model_params['min_elems'],balance=model_params['balance'],
                             prop=model_params['prop'])

            metrics = Metrics(data_dict['Y_test'],Y_pred)
            accuracy.append(metrics['accuracy'])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            ROC_AUC.append(metrics['ROC_AUC'])
            
        elif model == 'DT':
            DT = DecisionTreeClassifier()
            DT.fit(X_train, Y)
            Y_pred = DT.predict(X_test)
            accuracy.append(accuracy_score(y_true=Y_test, y_pred=Y_pred))
            precision.append(precision_score(y_true=Y_test, y_pred=Y_pred))
            recall.append(recall_score(y_true=Y_test, y_pred=Y_pred))
            ROC_AUC.append(roc_auc_score(y_true=Y_test, y_score=Y_pred))
            
        else:
            DT = KNeighborsClassifier()
            DT.fit(X_train, Y)
            Y_pred = DT.predict(X_test)
            accuracy.append(accuracy_score(y_true=Y_test, y_pred=Y_pred))
            precision.append(precision_score(y_true=Y_test, y_pred=Y_pred))
            recall.append(recall_score(y_true=Y_test, y_pred=Y_pred))
            ROC_AUC.append(roc_auc_score(y_true=Y_test, y_score=Y_pred))
             
    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'ROC_AUC':ROC_AUC}
    