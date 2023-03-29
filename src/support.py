import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import OrdinalEncoder




class Gestion_nulos:
    def  __init__(self, dataframe):
        self.df = dataframe

    def simple_imputer_method(self, column_list, strat_rep, missing):

        describe_ini= self.df.select_dtypes(include = np.number).describe()

        for column in column_list:

            imputer = SimpleImputer(strategy= strat_rep, missing_values=missing)
            imputer = imputer.fit(self.df[[column]])
            self.df[column] = imputer.transform(self.df[[column]])

        describe_fin= self.df.select_dtypes(include = np.number).describe()
        diferencia = describe_ini.subtract(describe_fin, fill_value=0)
        resultado= diferencia.loc['mean']

        return resultado


    def iterative_imputer_method(self):

        describe_ini= self.df.select_dtypes(include = np.number).describe()
        
        numericas = self.df.select_dtypes(include = np.number)
        imputer = IterativeImputer()
        imputer.fit(numericas)
        imputer.transform(numericas)
        numericas_trans = pd.DataFrame(imputer.transform(numericas), columns = numericas.columns)
        columnas = numericas_trans.columns
        self.df.drop(columnas, axis = 1, inplace = True)
        self.df[columnas] = numericas_trans

        describe_fin= self.df.select_dtypes(include = np.number).describe()
        diferencia = describe_ini.subtract(describe_fin, fill_value=0)
        resultado= diferencia.loc['mean']

        return resultado
    
    def knn_imputer_method(self, num_neighbors):

        describe_ini= self.df.select_dtypes(include = np.number).describe()

        numericas = self.df.select_dtypes(include = np.number)
        imputerKNN = KNNImputer(n_neighbors=num_neighbors)
        imputerKNN.fit(numericas)
        numericas_knn= imputerKNN.transform(numericas)
        df_knn_imputer = pd.DataFrame(numericas_knn, columns = numericas.columns)
        columnas_knn = df_knn_imputer.columns
        self.df.drop(df_knn_imputer, axis = 1, inplace = True)
        self.df[columnas_knn] = df_knn_imputer[columnas_knn]

        describe_fin= self.df.select_dtypes(include = np.number).describe()
        diferencia = describe_ini.subtract(describe_fin, fill_value=0)
        resultado= diferencia.loc['mean']

        return resultado
    
    
    
class Gestion_outliers:
    def  __init__(self, dataframe):
        self.df = dataframe

    def detectar_outliers_col_ifo(self, column):

        Q1 = np.nanpercentile(self.df[column], 25)
        Q3 = np.nanpercentile(self.df[column], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outliers_data = self.df[(self.df[column] < Q1 - outlier_step) | (self.df[column] > Q3 + outlier_step)]

        return outliers_data
    
    def detectar_outliers_index(self, column_list): 
    
        dicc_indices = {} 
    
        for col in column_list:
        
            Q1 = np.nanpercentile(self.df[col], 25)
            Q3 = np.nanpercentile(self.df[col], 75)
        
            IQR = Q3 - Q1
        
            outlier_step = 1.5 * IQR
        
            outliers_data = self.df[(self.df[col] < Q1 - outlier_step) | (self.df[col] > Q3 + outlier_step)]
        
        
            if outliers_data.shape[0] > 0:
        
                dicc_indices[col] = (list(outliers_data.index)) 
    
        return dicc_indices
    
    def eliminar_outliers(self, column_list): 
    
        dicc_indices = {} 
    
        for col in column_list:
        
            Q1 = np.nanpercentile(self.df[col], 25)
            Q3 = np.nanpercentile(self.df[col], 75)
        
            IQR = Q3 - Q1
        
            outlier_step = 1.5 * IQR
        
            outliers_data = self.df[(self.df[col] < Q1 - outlier_step) | (self.df[col] > Q3 + outlier_step)]
        
        
            if outliers_data.shape[0] > 0:
        
                dicc_indices[col] = (list(outliers_data.index)) 
        
        valores = list(dicc_indices.values())
        valores = [indice for sublista in valores for indice in sublista]
        valores = set(valores)
        df_sin_outliers2 = self.df.copy()
        df_eliminacion_outliers = df_sin_outliers2.drop(df_sin_outliers2.index[list(valores)] )
    
        return df_eliminacion_outliers
    
    
    def reemplazar_outliers(self, column_list, method): 
    
        dicc_indices = {} 
    
        for col in column_list:
        
            Q1 = np.nanpercentile(self.df[col], 25)
            Q3 = np.nanpercentile(self.df[col], 75)
        
            IQR = Q3 - Q1
        
            outlier_step = 1.5 * IQR
        
            outliers_data = self.df[(self.df[col] < Q1 - outlier_step) | (self.df[col] > Q3 + outlier_step)]
        
        
            if outliers_data.shape[0] > 0:
        
                dicc_indices[col] = (list(outliers_data.index)) 

        df_reemp_outliers = self.df.copy()

        for k, v in dicc_indices.items():
            if method== 'media':
                media = df_reemp_outliers[k].mean() 
                for i in v:
                    df_reemp_outliers.loc[i,k] = media
            else:
                mediana = df_reemp_outliers[k].median() 
                for i in v:
                    df_reemp_outliers.loc[i,k] = mediana
        
        return df_reemp_outliers

    
    
class Encoding:
    def  __init__(self, dataframe):
        self.df = dataframe

    def dummies_encoding(self, column_list):

        for column in column_list:

            dummies = pd.get_dummies(self.df[column], prefix_sep = "_", prefix = column, dtype = int)
            self.df[dummies.columns] = dummies
            self.df.drop([column], axis = 1, inplace = True)

        return self.df
    
    def one_hot_encoder(self, column_list):
    
        oh = OneHotEncoder()
    
        transformados = oh.fit_transform(self.df[column_list])
    
        oh_df = pd.DataFrame(transformados.toarray(), columns = oh.get_feature_names_out(), dtype = int)
    
        self.df[oh_df.columns] = oh_df
    
        self.df.drop(column_list, axis = 1, inplace = True)
    
        return self.df
    
    def ordinal_encoder(self, columna, orden_valores):
    
        ordinal = OrdinalEncoder(categories = [orden_valores], dtype = int)

        transformados_oe = ordinal.fit_transform(self.df[[columna]])
        oe_df = pd.DataFrame(transformados_oe)
        oe_df.columns = ordinal.feature_names_in_
        columna += "_oe"
        self.df[columna] = oe_df
     
        return self.df
    
    def porcentajes_variables_nonum(self, lista_columnas):

        for col in lista_columnas:
            porcentajes = (self.df[col].value_counts() / len(self.df)) * 100
            porcentajes_redondeados = round(porcentajes, 2)
            print(f"% valores columna: {col}")
            print(porcentajes_redondeados)
            print(f" ")

