import numpy as np

def activation_function(u, threshold = 0):
    return 1 if u >= threshold else 0

class Perceptron:
    def __init__(self):
        self.weights = []
    
    def fit(self, train_data, train_output, num_epochs = -1, shuffle = False, learning_rate = 0.1, sampling_range = 1.0):
        w = np.random.uniform(-sampling_range/2, sampling_range/2, len(train_data[0]) + 1)
        print("Pesos iniciais: {}".format(w))

        xs = np.asarray(train_data)
        xs = np.insert(xs, 0, -1, axis = 1)
        y = np.asarray(train_output)

        epoch = 0
        total_changes = 0

        changes = -1
        no_epochs = True
        if num_epochs > 0 :
            no_epochs = False
        while (changes != 0 and no_epochs) or epoch < num_epochs:
            changes = 0
            
            print("------ Época {} ------".format(epoch + 1))

            if shuffle:
                s = np.arange(xs.shape[0])
                np.random.shuffle(s)
                xs = xs[s]
                y = y[s]
            i = 0
            for x in xs:
                output = activation_function(np.dot(x, w))
                error = y[i] - output

                if error != 0:
                    w = w + learning_rate * error * x
                    print("Novos pesos: {}".format(w))
                    changes += 1

                i += 1
            
            print("Total de ajustes: {}".format(changes))
            epoch += 1
            total_changes += changes
            
        self.weights = w

        print("*********************")
        print("Total de épocas: {}".format(epoch))
        print("Total de ajustes de peso: {}".format(total_changes))
    
    def predict(self, input_data):
        xs = np.asarray(input_data)
        xs = np.insert(xs, 0, -1, axis = 1)

        ys = np.empty(0)

        for x in xs:
            output = activation_function(np.dot(x, self.weights))
            ys = np.append(ys, output)
        
        return ys