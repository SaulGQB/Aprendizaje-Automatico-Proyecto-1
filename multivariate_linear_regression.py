def batch_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de regresión lineal Batch Gradiente Descent
    batch_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epochs: numero máximo de iteraciones
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
    import random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
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
    normal_equation(X, Y, percent): 
    X: Dataframe de inputs 
    Y: Dataframe de outputs
    percent: % de datos que seran utilizados para el test (base 100)
    
    Return: theta, test_index, train_index, Y_predict
    
    test_index: indices de los valores utilizados para el test
    train_index: indices de los valores utilizados para el entrenamiento
    theta: valores correspondientes a theta_n
    Y_predict: valores de Y obtenidos de la predicción
    '''
    import numpy as np
    import pandas as pd
    import random as random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m)) 
    
    for indice in test_index:
        train_index.remove(indice)

    X_train = np.c_[X.iloc[train_index]]
    X_test = np.c_[X.iloc[test_index]]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento
    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_train
    
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    Y_predict = X_b_test @ theta
    
    return theta, test_index, train_index, Y_predict


def stochastic_gradient_descent(X, Y, eta, epochs, percent, batch_size):
    import numpy as np
    import pandas as pd
    import random as random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
         
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento
    
    theta = np.random.randn((X.shape[1] + 1), 1)
    
    J_log = np.zeros(epochs)
    
    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]

    for i in range(epochs):
        start = i * batch_size % X_b.shape[0]  
        end = min(start + batch_size, X_b.shape[0])
        idx = np.arange(start, end)
        batchX = X_b[idx]
        batchY = Y_train[idx]
        
        J_log[i] = (2 / m) * ((batchX  @theta - batchY)**2).sum()
        gradients = (1 / m) * (batchX.T @ (batchX  @ theta - batchY)) 
        theta = theta - eta * gradients
    
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    Y_predict = X_b_test @ theta
    
    return theta, test_index, train_index, Y_predict, J_log

def mini_batch_gradient_d(X, Y, eta, epochs, percent, b_size):
    import numpy as np
    import pandas as pd
    import random as random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))    
    for element in test_index:
        train_index.remove(element)
         
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento

    theta = np.random.randn((X.shape[1] + 1), 1)
    
    J_log = np.zeros(epochs)
    
    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]
    
    batch_size = b_size

    for i in range(epochs):
        mini_batches = create_mini_batches(X_b, Y_train, batch_size)
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            J_log[i] = (2 / m) * ((X_mini @theta - Y_mini)**2).sum()
            gradients = (1 / m) * (X_mini.T @ (X_mini @ theta - Y_mini)) 
            theta = theta - eta * gradients    
     
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    Y_predict = X_b_test @ theta
    
    return theta, test_index, train_index, Y_predict, J_log
       
def create_mini_batches(X, Y, batch_size):
    import numpy as np
    mini_batches = []
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0
  
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches
    
