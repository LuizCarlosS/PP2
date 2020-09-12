import numpy as np

def loadDataset(path):
    """
    Importa um array unidimensional de um arquivo binário, reformata em um array bidimensional,
    separa em outros dois arrays `X` e `Y` retorna-os.

    Parameters
    ----------
    path : string
        O caminho do arquivo que se deseja carregar.

    Returns
    -------
    X : ndarray.
        Um array de tuplas (x1,x2) que representa o conjunto de treino.
    Y : ndarray
        Um array unidimensional que representa as respostas corretas para cada tupla.
        `X` e `Y` têm o mesmo comprimento.
    """

    # Leitura do arquivo binário de dados
    data = np.fromfile(path)

    # `data` é um array unidimensional em que a cada 3 posições podemos formar uma tupla do tipo (x1,x2,y)
    # Onde x1 e x2 são pontos no plano e y é a classe (0 ou 1) a qual o ponto pertence

    # Portanto, criamos um array bidimensional em que cada linha possui uma tupla (que na verdade é um array mesmo)
    data = data.reshape(data.shape[0]//3, 3)
    print(data[0:5])

    # `X` contém todos os pontos (x1,x2)
    X = data[:,:-1]
    # `Y` contém as classes para cada ponto em `X`
    Y = data[:,-1]

    return X, Y