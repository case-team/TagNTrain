import tensorflow as tf
import numpy as np
from sklearn.preprocessing import quantile_transform


class ModelEnsemble:

    def __init__(self, model_names = None, location = "", model_type = -1, num_models = 0):
        self.model_list = []
        if(model_names == None):
            if(len(location) == 0 or num_models <=0):
                print("ModelEnsemble: Need to input model list or directory + num models")
                exit(1)
            self.model_names = []

            #print(location, location[-2:])
            if(num_models == 1 and location[-2:] == 'h5'): #single model inputed, not a directory
                self.model_names.append(location)

            else: #inputed a directory
                for i in range(num_models):
                    model_name = "model%i.h5" % i
                    self.model_names.append(location + model_name)
                    model = tf.keras.models.load_model(location + model_name)
                    self.model_list.append(model)

        self.num_models = len(self.model_names)
        self.model_type = model_type

    def predict(self, X, batch_size = 512):
        for i in range(self.num_models):
            #Ys = [quantile_transform(model.predict(X, batch_size = batch_size)) for model in self.model_list]
            Ys = [model.predict(X, batch_size = batch_size) for model in self.model_list]
        return np.average(Ys, axis=0).reshape(-1)
        

