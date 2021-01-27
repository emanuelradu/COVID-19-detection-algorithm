# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:45:27 2020
@author: bogdan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def uniform_input(df):
    # drop the rows which have the result field empty (~ 500 rows dropped)
    df.dropna(subset=['rezultat testare'], inplace=True)
    print(df.shape)
    df = df.reset_index(drop=True)
    
    df = df.fillna(value='None')
    # UNIFORMIZAREA DATELOR PENTRU SIMPTOME DECLARATE

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
    # deci functia asta primeste 2 dictionare, unul cu coloana din df pe care vrem sa inlocuiasca
    # iar la ca value este dat un regex cu forma cuvintelor de inlocuit
    # si al doilea dictionar reprezinta valuarea cu care vrem sa inluciasca

    # Am zis sa avem 3 categorii:'
    #       Asimptomaticii - care ori au Nan in excel, ori incepe cuvantul cu ASIM (case insesitive)
    #       Simptome specifice - descrierea contine alaturari de litere precum: tuse, feb (de la febra),
    #                          disp (dispnee), fat (fatigabilitate), dur (durere), gre (greata)
    #       simptome generale - celulele care nu s-au modificat in uram schumbarilor de mai devreme (aici am avut nevoie sa
    #        folosesc ! la inceputul primelor 2 categorii ca sa pot sa fac diferentierea, regex nu permite negarea de forma
    #        "sa nu contina ceva")

    #### simptome declarate ####
    # cei care au declarat asimptoamtic sau nu au declarat -> asimptomatic
    df = df.replace({'simptome declarate': r'^(A|a)(S|s)(I|i)(M|m).*'}, {'simptome declarate': '!ASIMPTOMATIC'}, regex=True)
    df = df.replace({'simptome declarate': 'None'}, {'simptome declarate': '!ASIMPTOMATIC'}, regex=True)

    # cei cu simptome specifice
    df = df.replace({'simptome declarate':  r'.*(((T|t)(U|u)(S|s)(E|e))|((F|f)(E|e)(B|b))|((D|d)(I|i)(S|s)(P|p))|((F|f)(A|A)(T|t))|((D|d)(U|u)(R|r))|((G|g)(R|r)(E|e)))+.*'},
                    {'simptome declarate': '!SIMPTOME SPECIFICE'}, regex=True)

    df = df.replace({'simptome declarate': r'^[^!].*'}, {'simptome declarate': '!SIMPTOME GENERALE'}, regex=True)


    #### simptome raportate la internare ####
    # cei care au raportat la internare asimptoamtic sau nu au declarat -> asimptomatic
    df = df.replace({'simptome raportate la internare': r'^(A|a)(S|s)(I|i)(M|m).*'}, {'simptome raportate la internare': '!ASIMPTOMATIC'}, regex=True)
    df = df.replace({'simptome raportate la internare': 'None'}, {'simptome raportate la internare': '!ASIMPTOMATIC'}, regex=True)

    # cei cu simptome specifice
    df = df.replace({'simptome raportate la internare':  r'.*(((T|t)(U|u)(S|s)(E|e))|((F|f)(E|e)(B|b))|((D|d)(I|i)(S|s)(P|p))|((F|f)(A|A)(T|t))|((D|d)(U|u)(R|r))|((G|g)(R|r)(E|e)))+.*'},
                    {'simptome raportate la internare': '!SIMPTOME SPECIFICE'}, regex=True)

    df = df.replace({'simptome raportate la internare': r'^[^!].*'}, {'simptome raportate la internare': '!SIMPTOME GENERALE'}, regex=True)


    #### diagnostic și semne de internare ####
    # cei care sunt suspecti sau confirmati de covid
    df = df.replace({'diagnostic și semne de internare': r'^.*((C|c)(O|o)((V|v)|((R|r)(O|o)(N|n)(A|a))))+.*'}, {'diagnostic și semne de internare': '!COVID'}, regex=True)

    # cei care au febra
    df = df.replace({'diagnostic și semne de internare': r'^.*((F|f)(((E|e)|(I|i))(B|b)(R|r))|((R|r)(I|i)(S|s)))+.*'}, {'diagnostic și semne de internare': '!FEBRA'}, regex=True)

    #cei care au probleme pulmonare
    df = df.replace({'diagnostic și semne de internare':
        r'^.*((P|p)(N|n)(E|e)(U|u)(M|m)(O|o))|((B|b)(R|r)(O|o)(N|n)(H|h)(O|o))|((R|r)(E|e)(S|s)(P|p)(I|i)(R|r))|((T|t)(R|r)(O|o)(M|m)(B|b)(O|o))|((P|p)(U|u)(L|l)(M|m)(O|o))+.*'
        }, {'diagnostic și semne de internare': '!PLAMANI'}, regex=True)

    # cei despre care nu avem mai multe informatii
    df = df.replace({'diagnostic și semne de internare': 'None'}, {'diagnostic și semne de internare': '!NEDETERMINAT'}, regex=True)

    # alte diagnostice irelevante
    df = df.replace({'diagnostic și semne de internare': r'^[^!].*'}, {'diagnostic și semne de internare': '!ALTELE'}, regex=True)

    #### istoric de calatorie ####
    df = df.replace({'istoric de călătorie':  r'.*(((N|n)(E|e)(A|a)(G|g))|0|(N|n)(U|u))+.*'},
                    {'istoric de călătorie': '!NU'}, regex=True)
    df = df.replace({'istoric de călătorie': r'^(D|d)(A|a).*'}, {'istoric de călătorie': '!DA'}, regex=True)
    df = df.replace({'istoric de călătorie': 'None'}, {'istoric de călătorie': '!INEXISTENT'}, regex=True)
    df = df.replace({'istoric de călătorie': r'^[^!].*'}, {'istoric de călătorie': '!DA'}, regex=True)

    #### mijloace de transport folosite ####
    df = df.replace({'mijloace de transport folosite':  r'.*(((N|n)(E|e)(A|a)(G|g))|0|(N|n)(U|u))+.*'},
                    {'mijloace de transport folosite': '!NU'}, regex=True)
    df = df.replace({'mijloace de transport folosite': r'^(D|d)(A|a).*'}, {'mijloace de transport folosite': '!DA'}, regex=True)
    df = df.replace({'mijloace de transport folosite': 'None'}, {'mijloace de transport folosite': '!INEXISTENT'}, regex=True)
    df = df.replace({'mijloace de transport folosite': r'^[^!].*'}, {'mijloace de transport folosite': '!DA'}, regex=True)

    #### confirmare contact cu o persoană infectată ####
    df = df.replace({'confirmare contact cu o persoană infectată':  r'.*(((N|n)(E|e)(A|a)(G|g))|0|(N|n)(U|u))+.*'},
                    {'confirmare contact cu o persoană infectată': '!NU'}, regex=True)
    df = df.replace({'confirmare contact cu o persoană infectată': r'^(D|d)(A|a).*'}, {'confirmare contact cu o persoană infectată': '!DA'}, regex=True)
    df = df.replace({'confirmare contact cu o persoană infectată': 'None'}, {'confirmare contact cu o persoană infectată': '!NODATA'}, regex=True)
    df = df.replace({'confirmare contact cu o persoană infectată': r'^[^!].*'}, {'confirmare contact cu o persoană infectată': '!DA'}, regex=True)

    #### rezultat testare ####
    df = df.replace({'rezultat testare': 'NEGATIV'}, {'rezultat testare': '!NEGATIV'}, regex=True)
    df = df.replace({'rezultat testare': 'POZITIV'}, {'rezultat testare': '!POZITIV'}, regex=True)
    df = df.replace({'rezultat testare':  r'^[^!].*'}, {'rezultat testare': '!NEGATIV'}, regex=True)
    

    return df

def prepare():
    columns = ['instituția sursă', 'sex', 'vârstă', 'dată debut simptome declarate',
               'simptome declarate', 'dată internare', 'simptome raportate la internare',
               'diagnostic și semne de internare', 'istoric de călătorie',
               'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
               'data rezultat testare', 'rezultat testare']
    
    # our file is larger then the actual nr of rows and columns
    # the nrows and columns should be given as input intel
    # morever default enconding does not work, because the dataset is with diacritics
    df = pd.read_csv('mps.dataset.csv', names=columns, encoding = 'utf-8', nrows=6444)
    print(df.shape)
    
    
    df = uniform_input(df)
    
    # Get names of indexes for which column Age has value 30
    indexNames = df[ df['rezultat testare'] == '!TO_DROP' ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)
    df = df.reset_index(drop=True)
    
    
    print(df['rezultat testare'].value_counts())
    
    
    # Randomize the datset
    np.random.seed(5)
    l = list(df.index)
    np.random.shuffle(l)
    df = df.iloc[l]
    
    # Training = 70% of the data
    # Validation = 30% of the data
    
    rows = df.shape[0]
    train = int(.7 * rows)
    test = rows - train
    
    columns = ['simptome declarate', 'simptome raportate la internare',
               'diagnostic și semne de internare', 'istoric de călătorie',
               'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
               'rezultat testare']
    
    # Write Training Set
    df[:train].to_csv('covid_train.csv'
                              ,index=False,index_label='Row',header=False
                              ,columns=columns)
    
    # Write Validation Set
    df[train:].to_csv('covid_validation.csv'
                              ,index=False,index_label='Row',header=False
                              ,columns=columns)



def prepare_test(filename):
    columns = ['instituția sursă', 'sex', 'vârstă', 'dată debut simptome declarate',
           'simptome declarate', 'dată internare', 'simptome raportate la internare',
           'diagnostic și semne de internare', 'istoric de călătorie',
           'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
           'data rezultat testare', 'rezultat testare']

    # our file is larger then the actual nr of rows and columns
    # the nrows and columns should be given as input intel
    # morever default enconding does not work, because the dataset is with diacritics
    df = pd.read_csv(filename, names=columns, encoding = 'utf-8')
    print(df.shape)
    
    
    df = uniform_input(df)
    
    indexNames = df[ df['rezultat testare'] == '!TO_DROP' ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)
    df = df.reset_index(drop=True)
    
    
    print(df['rezultat testare'].value_counts())
    
    
    # Randomize the datset
    np.random.seed(5)
    l = list(df.index)
    np.random.shuffle(l)
    df = df.iloc[l]
    
    
    columns = ['simptome declarate', 'simptome raportate la internare',
               'diagnostic și semne de internare', 'istoric de călătorie',
               'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
               'rezultat testare']
    
    df[:].to_csv('test.csv'
                              ,index=False,index_label='Row',header=False
                              ,columns=columns)
    
if __name__ == '__main__':
    prepare()