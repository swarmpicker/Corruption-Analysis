from sklearn.neural_network import MLPRegressor
import numpy as np

X = np.loadtxt(r'D:\Documents\ITC-NN\ChinaIPR\Heritage Scaled IVs.csv', delimiter=',')
y = np.loadtxt(r'D:\Documents\ITC-NN\ChinaIPR\Heritage Corruption DV.csv', delimiter=',')

print("Inputs")
print(X)
                      
reg=MLPRegressor(activation='logistic', 
       alpha=1e-05, 
       batch_size='auto',
       beta_1=0.9, beta_2=0.999, 
       early_stopping=False,
       epsilon=1e-08, 
       hidden_layer_sizes=(8,8), 
       learning_rate='constant',
       learning_rate_init=0.001, 
       max_iter=5000, 
       momentum=0.9,
       nesterovs_momentum=True, 
       power_t=0.5, 
     #  random_state=None, #see how this works 
       shuffle=True,
       solver='lbfgs',
       tol=0.0001, 
       validation_fraction=0.1, 
       verbose=False,
       warm_start=False)

reg.fit(X, y)     

print 'Number of layers: %s. Number of outputs: %s' % (reg.n_layers_, reg.n_outputs_)

print('Variance explained: %.3f' % reg.score(X, y))
print("Residual sum of squares: %.3f"
       % np.mean((reg.predict(X) - y) ** 2))
y_predict = reg.predict(X)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print("Predicting y:")
print( y_predict)
print("Target")
print(y)
print("Weights")
print(reg.coefs_)
