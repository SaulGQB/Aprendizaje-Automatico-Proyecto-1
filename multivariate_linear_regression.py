def batch_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de regresión lineal Batch Gradiente Descent
    batch_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epocs: numero máximo de iteraciones
    percent: % de datos que seran utilizados para el test (base 100)
    
    ------------------------------------
    Return:
    In order: theta, test_index, train_index, Y_predict, J_log
    
    theta: valores correspondientes a theta_n
    test_index: data test index
    train_index: data training index
    Y_predict: Y predict values
    J_log: errores por numero de epoca
    '''
    import numpy as np
    import pandas as pd
    import random as random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = np.c_[X.iloc[train_index]]
    X_test = np.c_[X.iloc[test_index]]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento
    
    theta = np.random.randn((X.shape[1] + 1), 1)
    
    J_log = np.zeros(epochs)
    
    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]

    for i in range(epochs):
        J_log[i] = (2 / m) * ((X_b@theta - Y_train)**2).sum()
        gradients = (1 / m) * (X_b.T @ (X_b @ theta - Y_train)) 
        theta = theta - eta * gradients
    
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    Y_predict = X_b_test @ theta
    
    return theta, test_index, train_index, Y_predict, J_log


def normal_equation(X, Y, percent):
    '''Esta función sirve para utilizar el método de regresión lineal con ecuación normal
    normal_equation(X, Y): 
    X: Matriz columna de inputs 
    Y: Matriz columna de outputs
    percent: % de datos que seran utilizados para el test (base 100)
    
    Return: indices_test, indices_train, theta, Y_predict
    
    indices_test: indices de los valores utilizados para el test
    indices_train: indices de los valores utilizados para el entrenamiento
    theta: valores correspondientes a theta_n
    Y_predict: valores de Y obtenidos de la predicción
    '''
    import numpy as np
    import pandas as pd
    import random as random
    
    m = len(X)
    indices_test = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    indices_train = list(np.arange(0, m))
    
    for indice in indices_test:
        indices_train.remove(indice)

    X_train = np.c_[X.iloc[indices_train]]
    X_test = np.c_[X.iloc[indices_test]]
    Y_train = np.c_[Y.iloc[indices_train]]
    Y_test = np.c_[Y.iloc[indices_test]]
    
    # Entrenamiento
    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_train
    
    # test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    Y_predict = X_b_test @ theta
    
    return indices_test, indices_train, theta, Y_predict, Y_test
