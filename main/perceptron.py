import numpy as np

"""
=========================================================
perceptron.py
=========================================================

Provides
  1. Um neurônio Perceptron de Rosenblatt
  2. Função de ativação
  3. Treinamento e predição de resultados

=========================================================
How to use 
=========================================================
Em seu projeto, importe o arquivo desta maneira:

    >>> from perceptron import Perceptron

Crie uma instância de perceptron: 

    >>> p =  Perceptron()

Utilize os métodos:

    >>> p.fit(X, Y);
    >>> p.predict(X);

==================== =========================================================
Utility functions
==============================================================================
fit                   Ajusta o neurônio de acordo com os parâmetros de entrada.
predict               Faz uma predição em cima dos dados de entrada.
activation_function   Função de ativação do neurônio.
==================== =========================================================

"""

def activation_function(u, threshold = 0):
    """
    Retorna a ativação ou não ativação do neurônio.

    Parameters
    ----------
    u : float
        Soma de todos as entradas * pesos do neurônio.

    threshold : int
        Limiar de ativação do neurônio. Valor padrão: 0.

    Returns
    -------
    out : int.
        1 para neurônio ativado. 0 caso contrário.
    """

    return 1 if u >= threshold else 0



class Perceptron:
    """
    Uma classe usada para representar um neurônio de Rosenblatt.

    ...

    Attributes
    ----------

    Methods
    -------
    fit(train_data, train_output, num_epochs = -1, shuffle = False, learning_rate = 0.1, sampling_range = 1.0)
        Ajusta o neurônio de acordo com os parâmetros de entrada.

    predict(input_data)
        Faz uma predição em cima dos dados de entrada.
    """

    def __init__(self):
        self.weights = []
        self.epoch_changes = []
    
    def fit(self, train_data, train_output, num_epochs = -1, shuffle = False, learning_rate = 0.1, sampling_range = 1.0, verbose = True):
        """
        Ajusta o neurônio de acordo com os parâmetros de entrada. O ajustamento é feito seguindo 
        o algoritmo apresentado em aula.

        Se os argumentos `num_epochs`, `shuffle`,  `learning_rate` ou `sampling_range` não são fornecidos, 
        os valores padrão são utilizados.
        
        Parameters
        ----------
        train_data : ndarray
            Um conjunto de valores reais em tuplas que representam um ponto no plano cartesiano.

        train_output : ndarray
            Um conjunto de valores inteiros que representam a classe de um ponto no plano.
            `train_data` e `train_output` devem ter o mesmo comprimento.
        
        num_epochs : int, optional
            A quantidade de épocas que se deseja ajustar o neurônio.

        shuffle : bool, optional
            Utilizado para embaralhar `train_data` quando necessário.

        learning_rate : float (0, 1], optional
            Taxa de aprendizado do neurônio. O quão rápido ou lento se deseja (tentar) chegar à convergência.

        sampling_range : float, optional
            O intervalo que se deseja gerar uma variável uniforme aleatória.
	
        verbose : bool, optional
            Aciona ou não os prints durante o treinamento.

        Prints
        ------
        Pesos iniciais
            Os pesos iniciais utilizados para o treinamento.

        Total de épocas
            A quantidade total de épocas passadas até convergir ou atingir o máximo de `num_epochs`.

        Total de ajustes de pesos
            A quantidade de vezes que os pesos foram ajustados até convergir ou atingir o máximo de `num_epochs`.
        """

        # O vetor inicial de pesos deve ter seus valores inicializados conforme 
        # uma variável aleatória de distribuição uniforme no intervalo
        w = np.random.uniform(-sampling_range/2, sampling_range/2, len(train_data[0]) + 1)
        if verbose:
            print("Pesos iniciais: {}".format(w))

        # Garante que os arrays de trabalho são numpy
        xs = np.asarray(train_data)
        y = np.asarray(train_output)

        # Insere o valor de bias em cada tupla do conjunto
        xs = np.insert(xs, 0, -1, axis = 1)

        epoch = 0
        total_changes = 0

        changes = -1 # armazena quantidade atual de mudanças
        no_epochs = True

        # Lógica para o treinamento rodar enquanto houver ajustamento de pesos 
        # ou enquanto houver `num_epochs` restantes.
        if num_epochs > 0 :
            no_epochs = False

        while (changes != 0 and no_epochs) or epoch < num_epochs:
            changes = 0
            
            if verbose:
                print("------ Época {} ------".format(epoch + 1))

            # Para quando os exemplos devem ser aleatoriamente divididos
            if shuffle:
                s = np.arange(xs.shape[0])
                np.random.shuffle(s)
                xs = xs[s]
                y = y[s]

            i = 0
            # Aplique a regra de atualização até que `error` == 0 para todos os `x` elementos de `xs`.
            for x in xs: 
                output = activation_function(np.dot(x, w))
                error = y[i] - output

                if error != 0:
                    # Regra de atualização dos pesos
                    w = w + learning_rate * error * x
                    # Sempre que o vetor de pesos for ajustado, este deve ser impresso
                    if verbose:
                        print("Novos pesos: {}".format(w))
                    changes += 1

                i += 1

            if(changes > 0):
            	self.epoch_changes.append(changes)
            	
            # A cada época deve ser indicado o número de ajustes feitos no vetor de pesos
            if verbose:
                print("Total de ajustes: {}".format(changes))
            epoch += 1
            total_changes += changes
            
        self.weights = w

        # Ao final, deve-se imprimir:
        ## (a) O número total de ajustes no vetor de pesos;
        ## (b) O número de épocas até a convergência;
        if verbose:
            print("*********************")
            print("Total de épocas: {}".format(epoch))
            print("Total de ajustes de peso: {}".format(total_changes))

    
    def predict(self, input_data):
        """
        Faz uma predição de classe em cima dos dados de entrada.
        
        Parameters
        ----------
        input_data : ndarray
            Um conjunto de valores reais em tuplas que representam um ponto no plano cartesiano.

        Returns
        ------
        ys : ndarray
            Um array de valores preditos que representam a classe de um ponto de `input_data`.
        """

        xs = np.asarray(input_data)

        # Inserção de bias em cada tupla
        xs = np.insert(xs, 0, -1, axis = 1)

        ys = np.empty(0)

        # Para todo ponto do plano, some as entradas * pesos e passe pela função de ativação
        for x in xs:
            output = activation_function(np.dot(x, self.weights))
            ys = np.append(ys, output)
        
        return ys