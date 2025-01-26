import tensorflow as tf
import os

class DeepQNetwork(object):
    def __init__(self, state_size, action_size, lr= 0.001, chkpt_dir= 'tmp/dqn'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.model = self._build_model()
        self.saver = tf.train.Checkpoint(model= self.model)
        self.checkpoint_file = os.path.join(chkpt_dir, 'dqn_model.ckpt')
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape= (self.state_size,)),
            tf.keras.layers.Dense(64, activation= 'relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.3),
            tf.keras.layers.Dense(48, activation= 'relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(.3),
            tf.keras.layers.Dense(32, activation= 'relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss="huber_loss", 
            optimizer= tf.keras.optimizers.Adam(learning_rate= self.lr))
        return model
      
    def predict(self, state):
        return self.model(state).numpy()

    
    def fit(self, state, target, epochs= 1, verbose= 0):
        return self.model.fit(state, target, epochs=epochs, verbose=verbose)
    
    def load_checkpoint(self):
        print("... Loading checkpoint ...")
        self.model.load_weights(self.checkpoint_file)

    def save_checkpoint(self):
        print("... Saving checkpoint ...")
        self.model.save_weights(self.checkpoint_file)

        

