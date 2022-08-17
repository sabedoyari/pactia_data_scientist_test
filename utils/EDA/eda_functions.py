import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context('notebook')
sns.set_style('white')

def plot_missing_values(data):
    
    data_nan = data.isna().sum()
    data_nan = data_nan.sort_values(ascending = True)
    data_nan = data_nan[data_nan.values != 0]
    print(f'\nHay {data_nan.values.sum()} valores faltantes, NaN, nulls', end = '\n' * 2)

    plt.figure(figsize = (5, 1.25*data_nan.shape[0]))
    plt.barh(y = data_nan.index,
          width = data_nan.values,
          color = 'lightblue',
          alpha = 0.5,
          edgecolor = 'darkblue',
          linewidth = 1)
    plt.xlabel('Cantidad de valores faltantes')
    plt.ylabel('Atributos')
    plt.xticks(rotation = 90)
    # plt.show()
    
    print(data.isna().sum().sort_values(ascending = False))
    
def plot_univariate_attribute(data):
    
    """Esta función simplifica la visualización de todos los atributos que componen
    el data set pasado como argumento, separando las variables numéricas de las categóricas.
    """
    
    numerical_features = []
    categorical_features = []
    features = data.columns
    
    for feature in features:
        if data[feature].dtype == 'int64' or data[feature].dtype == 'float64':
            numerical_features.append(feature)
        elif data[feature].dtype == np.object:
            categorical_features.append(feature)
    
    #  Gráficas para las variables numéricas:
    print('Variables numéricas:\n')
    if len(numerical_features) == 1:
      atr = numerical_features[0]
      fig, ax = plt.subplots(len(numerical_features), ncols = 2, figsize = (9, 2*len(numerical_features)))

      ax[0].boxplot(data[atr], vert = False, showmeans = True, widths = 0.5)
      ax[1].boxplot(data[atr], vert = False, showmeans = True, widths = 0.5, showfliers = False)
      ax[0].set_title(atr, fontsize = 10)
      ax[1].set_title(atr + ' sin extremos', fontsize = 10)
      ax[0].set_ylabel(atr, fontsize = 10)

      plt.tight_layout()
    #   plt.show()
    else:
      fig, ax = plt.subplots(len(numerical_features), ncols = 2, figsize = (12, 2*len(numerical_features)))
      
      for i, atr in enumerate(numerical_features):
          ax[i][0].boxplot(data[atr], vert = False, showmeans = True, widths = 0.5)
          ax[i][1].boxplot(data[atr], vert = False, showmeans = True, widths = 0.5, showfliers = False)
          ax[i][0].set_title(atr, fontsize = 10)
          ax[i][1].set_title(atr + ' sin extremos', fontsize = 10)
          ax[i][0].set_ylabel(atr, fontsize = 10)
      
      plt.tight_layout()
    #   plt.show()
    
    #  Gráficas para las variables categóricas:
    print('Variables categóricas:\n')
    for i, atr in enumerate(categorical_features):
        sns.catplot(data = data, x = atr, 
                    kind = 'count',
                    height = 3, aspect = 5,
                    order = data[atr].value_counts(ascending = False).index)
        plt.xticks(rotation = 'vertical')
        plt.tight_layout
        # plt.show()
        
def plot_duplicated_records(data):
    Dup_prop = data[data.duplicated(keep = False)]
    Lista_Dup_prop = Dup_prop.groupby(Dup_prop.columns.tolist()).apply(lambda x: x.index.tolist()).values.tolist()
    print('En el DataSet se identificaron {} registros duplicados.'.format(len(Lista_Dup_prop)), end = '\n' * 2)
    
    if len(Lista_Dup_prop) != 0:
        plt.figure(figsize = (5, 5))
        plt.hist(x = [len(x) for x in Lista_Dup_prop], color = 'g', alpha = 0.5, edgecolor = 'darkgreen', linewidth = 1)
        plt.xlabel('Cantidad de repeticiones por registro', fontsize = 18)
        plt.ylabel('Frecuencia', fontsize = 18)
        #plt.title('Histograma de frecuencias según la cantidad de repeticiones de los valores duplicados')
        # plt.show()

def starting_descriptive_analysis(data):
    
    print("5 primeras filas del data set:\n\n")
    display(data.head())
    
    print("\nColumnas del data set e información preliminar:\n\n")
    display(data.info())
    
    print("\nValores faltantes por atributo:\n\n")
    plot_missing_values(data)