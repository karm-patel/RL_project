from config import config
import numpy as np
import tensorflow as tf

class Environment:
    def __init__(self):
        (X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        self.data_x = X_train.reshape(X_train.shape[0],-1)
        self.data_y = y_train
        # data = np.concatenate((x_train, y_train.reshape(-1,1)), axis=1)
        
        
        # self.data_x = data[:, 0:-1].astype('float32')
        # self.data_y = y_train.astype('int32')

        self.costs = -1*np.ones(self.data_x.shape[1])
        self.data_len = len(self.data_x)

        self.x_actual = None
        self.y = None
        self.mask = np.zeros( ( config.FEATURE_DIM), dtype=np.float32 )
        self.x_bar = np.zeros( ( config.FEATURE_DIM), dtype=np.float32 )
        self.state = (self.x_actual, self.y ,self.x_bar, self.mask)

    def _get_state(self, x_bar, mask):
        return (self.x_actual, self.y , x_bar, mask)
        
    def reset(self):
        sample_ind = np.random.choice(np.arange(0,self.data_len))
        self.x_actual = self.data_x[sample_ind]
        self.y = self.data_y[sample_ind]
        self.mask = np.zeros( ( config.FEATURE_DIM), dtype=np.float32 )
        self.x_bar = np.zeros( ( config.FEATURE_DIM), dtype=np.float32 )
        return self._get_state(self.x_bar, self.mask)

    def step(self, action):
        done = 0
        if action >= config.CLASSES:
            # take feature
            f_ind = action - config.CLASSES
            reward = self.costs[f_ind] * config.LAMBDA
            self.x_bar[f_ind] = self.x_actual[f_ind]
            self.mask[f_ind] = 1
            next_state = self._get_state(self.x_bar, self.mask)

        else:
            # classify action 
            done = 1
            y_pred = action
            reward = config.REWARD_CORRECT if y_pred == self.y else config.REWARD_INCORRECT 
            next_state = None
            
        return next_state, reward, done
        


