import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Application de détection de fraude de carte de Crédit")
#Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques de la transaction")

def client_caract_entree():
    Time=st.sidebar.slider('Temps écoulé depuis la première transaction',0.0,171000.0,0.1)
    Amount=st.sidebar.slider('Montant de la transaction en dollar',0.0,3000.0,1.0)
    V1=st.sidebar.slider('Valeur de V1',-60.0,10.0,0.1)
    V2=st.sidebar.slider('Valeur de V2',-80.0,30.0,0.1)
    V3=st.sidebar.slider('Valeur de V3',-50.0,10.0,0.1)
    V4=st.sidebar.slider('Valeur de V4',-10.0,20.0,0.1)
    V5=st.sidebar.slider('Valeur de V5',-120.0,40.0,0.1)
    V6=st.sidebar.slider('Valeur de V6',-30.0,80.0,0.1)
    V7=st.sidebar.slider('Valeur de V7',-50.0,125.0,0.1)
    V8=st.sidebar.slider('Valeur de V8',-75.0,25.0,0.1)
    V9=st.sidebar.slider('Valeur de V9',-15.0,20.0,0.1)
    V10=st.sidebar.slider('Valeur de V10',-10.0,10.0,0.1)
    V11=st.sidebar.slider('Valeur de V11',-10.0,10.0,0.1)
    V12=st.sidebar.slider('Valeur de V12',-10.0,10.0,0.1)
    V13=st.sidebar.slider('Valeur de V13',-10.0,10.0,0.1)
    V14=st.sidebar.slider('Valeur de V14',-10.0,10.0,0.1)
    V15=st.sidebar.slider('Valeur de V15',-10.0,10.0,0.1)
    V16=st.sidebar.slider('Valeur de V16',-10.0,10.0,0.1)
    V17=st.sidebar.slider('Valeur de V17',-10.0,10.0,0.1)
    V18=st.sidebar.slider('Valeur de V18',-10.0,10.0,0.1)
    V19=st.sidebar.slider('Valeur de V19',-10.0,10.0,0.1)
    V20=st.sidebar.slider('Valeur de V20',-10.0,10.0,0.1)
    V21=st.sidebar.slider('Valeur de V21',-10.0,10.0,0.1)
    V22=st.sidebar.slider('Valeur de V22',-10.0,10.0,0.1)
    V23=st.sidebar.slider('Valeur de V23',-10.0,10.0,0.1)
    V24=st.sidebar.slider('Valeur de V24',-10.0,10.0,0.1)
    V25=st.sidebar.slider('Valeur de V25',-10.0,10.0,0.1)
    V26=st.sidebar.slider('Valeur de V26',-10.0,10.0,0.1)
    V27=st.sidebar.slider('Valeur de V27',-10.0,10.0,0.1)
    V28=st.sidebar.slider('Valeur de V28',-10.0,10.0,0.1)
    data={
    'Time':Time,
    'V1':V1,
    'V2':V2,
    'V3':V3,
    'V4':V4,
    'V5':V5,
    'V6':V6,
    'V7':V7,
    'V8':V8,
    'V9':V9,
    'V10':V10,
    'V11':V11,
    'V12':V12,
    'V13':V13,
    'V14':V14,
    'V15':V15,
    'V16':V16,
    'V17':V17,
    'V18':V18,
    'V19':V19,
    'V20':V20,
    'V21':V21,
    'V22':V22,
    'V23':V23,
    'V24':V24,
    'V25':V25,
    'V26':V26,
    'V27':V27,
    'V28':V28,
    'Amount':Amount
    }

    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_caract_entree()



#Transformer les données d'entrée en données adaptées à notre modèle
#importer la base de données

df=pd.read_csv('creditcard.csv')
credit_input=df.drop(columns='Class')
donnee_entree=pd.concat([input_df,credit_input],axis=0)

#prendre uniquement la premiere ligne
donnee_entree=donnee_entree[:1]



#importer le modèle
load_model=pickle.load(open('modelF.pkl','rb'))


#appliquer le modèle sur le profil d'entrée
prevision=load_model.predict(donnee_entree)

#afficher les données transformées
st.subheader('Les caracteristiques transformées')
donnee_entree = donnee_entree.loc[:,~donnee_entree.columns.duplicated()]
st.write(donnee_entree)

#résultat de la prédiction

st.subheader('Résultat de la prévision')
st.write(prevision)
if prevision==0:
    st.write('transaction normale')
else:
    st.write('transaction frauduleuse')
#fin de l'application(Réalisé par Kossi Robert MESSAN)
