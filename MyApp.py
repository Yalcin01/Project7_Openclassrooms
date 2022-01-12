import streamlit as st
import pandas as pd
import datetime
import requests
import json
import pickle
import shap
import streamlit.components.v1 as components
import matplotlib as plt
from codes_P7 import *
import numpy
from pylab import *
import pylab as pl
import seaborn as sns


# url pouvant servir :
url_post = 'http://localhost:5000/post'

url_streamlit = 'http://localhost:8501/'

# url de l'api :
url="http://127.0.0.1:5000/api"


# début du code 
## message de bienvenue :
st.title("""
Bonjour et bienvenue
""")

# téléchargement des modèles de prédictions :
model = pickle.load(open('Modele/model_LGBM_pickle.pkl', 'rb'))
LGBM_best = pickle.load(open('Modele/LGBM_best.pkl', 'rb'))

# chargement des databases :
## databases avec les valeurs brutes, initiales :
app_train = pd.read_csv('data/app_train_sample.csv')  # read a CSV file inside the 'data" folder next to 'MyApp.py'
app_test = pd.read_csv('data/app_test_sample.csv')

app_train_2 = app_train.drop(columns = ['TARGET'])

## databases centrées réduites :
final_datas = pd.read_csv('data/Final_datas_sample.csv')
final_datas_test = pd.read_csv('data/Final_datas_test_sample.csv')

# ne pas afficher les messages d'erreur de version
st.set_option('deprecation.showPyplotGlobalUse', False)

# paramétrage de shap :
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# définition de la variable features (colonnes utiles pour la modélisation) :
features = list(final_datas.columns)
del features[0]
del features[-1]

# # définition de la variable features_affichage (colonnes à afficher sur le dashboard) :
features_affichage = list(app_train_2.columns)
del features_affichage[0]
del features_affichage[-1]

features_affichage_2 = list(app_train.columns)
del features_affichage_2[0]


# selectbox initiale qui oriente les choix de l'utilisateur :
choix = st.selectbox("Que souhaitez-vous faire ?",
     ('','Estimer un crédit', 'Consulter la base de données'))


if choix=='Estimer un crédit':
    st.write('Vous souhaitez estimer un crédit.')
    
    option = st.selectbox('Êtes-vous déjà client ?',
        ('','Oui', 'Non'))

    if option!='':
        st.write('Votre choix :', option)
        

    # Cas nouveau client, utilisation base de données Test :
    if option=='Non':

        z = st.selectbox('Quel est votre Identifiant ?',
                     app_test['SK_ID_CURR'])
        
        st.write(app_test[features_affichage])

        if z!=0:
        
            st.write('Votre ID est', int(z), '. Bienvenue !')
            st.write('Voici vos informations :', (app_test[app_test['SK_ID_CURR']==z])[features_affichage])  # visualize my dataframe in the Streamlit app
        
            client_dict = final_datas_test[final_datas_test['SK_ID_CURR']==z][features].to_dict('records')
                 
        
            if st.button('Votre crédit est-il accordé ?'):
                
                # requêter l'API :
                data=json.dumps(client_dict[0])    
                r=requests.post(url,data)           
                print(r.json())            
                # prediction : résultat de la prédiciton du modèle Light GBM via l'API
                prediction = r.json()
                                
                if prediction == 0:
                    st.success('Féliciations, votre crédit est accordé !')
                if prediction == 1:
                    st.error("Nous sommes désolés, votre crédit n'est pas accordé.")
                    
                 
                # affichage du shap : 
                data_for_prediction = final_datas_test[final_datas_test['SK_ID_CURR']==z][features] # use 1 row of data here. Could use multiple rows if desired

                # Create object that can calculate shap values
                explainer = shap.TreeExplainer(LGBM_best)

                # Calculate Shap values
                shap_values = explainer.shap_values(data_for_prediction)

                #shap.initjs()
                st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)) 

                # caractéristiques client :
                amt_credit_client = app_test[app_test['SK_ID_CURR']== z]['AMT_CREDIT'].values
                amt_income_client = app_test[app_test['SK_ID_CURR']== z]['AMT_INCOME_TOTAL'].values

                # Affichage des graphes :
                pl.subplot(2, 1, 1)                
                ax1 = sns.distplot(app_test['AMT_CREDIT'])
                ax1.set_xlabel('Amount')
                ax1.set_title('AMT_CREDIT')
                ax1.get_xaxis().get_major_formatter().set_scientific(False)
                ax1.get_yaxis().get_major_formatter().set_scientific(False)
                plt.axvline(amt_credit_client, color='red')
                
                pl.subplot(2, 1, 2)
                ax2 = sns.distplot(app_test['AMT_INCOME_TOTAL'])
                ax2.set_xlabel('Amount')
                ax2.set_title('AMT_INCOME_TOTAL')
                ax2.get_xaxis().get_major_formatter().set_scientific(False)
                ax2.get_yaxis().get_major_formatter().set_scientific(False)
                plt.axvline(amt_income_client, color='red')
                
                pl.tight_layout()
                st.pyplot(pl) 
                
                st.write("Les courbes rouges représentent vos caractéristiques.")
            
    # Cas client connu, utilisation base de données Train :        
    if option=='Oui': 
        
        z = st.selectbox('Quel est votre Identifiant ?',
                     app_train_2['SK_ID_CURR'])
        
        st.write(app_train_2[features_affichage])

        if z!=0:
        
            st.write('Votre ID est', int(z), '. Bienvenue !')
            st.write('Voici vos informations :', (app_train_2[app_train_2['SK_ID_CURR']==z])[features_affichage])  # visualize my dataframe in the Streamlit app
        
            client_values = final_datas[final_datas['SK_ID_CURR']==z][features].values
                 
        
            if st.button('Votre crédit est-il accordé ?'):
                # on regarde la valeur de la colonne TARGET du client :
                if app_train[app_train['SK_ID_CURR']== z]['TARGET'].values == 0:
                    st.success('Féliciations, votre crédit est accordé !')
                if app_train[app_train['SK_ID_CURR']== z]['TARGET'].values == 1:
                    st.error("Nous sommes désolés, votre crédit n'est pas accordé.")

                # caractéristiques client :
                amt_credit_client = app_train[app_train['SK_ID_CURR']== z]['AMT_CREDIT'].values
                amt_income_client = app_train[app_train['SK_ID_CURR']== z]['AMT_INCOME_TOTAL'].values
                
                
                # Affichage des graphes :
                pl.subplot(2, 1, 1)                
                ax1 = sns.distplot(app_train['AMT_CREDIT'])
                ax1.set_xlabel('Amount')
                ax1.set_title('AMT_CREDIT')
                ax1.get_xaxis().get_major_formatter().set_scientific(False)
                ax1.get_yaxis().get_major_formatter().set_scientific(False)
                plt.axvline(amt_credit_client, color='red')

                pl.subplot(2, 1, 2)
                ax2 = sns.distplot(app_train['AMT_INCOME_TOTAL'])
                ax2.set_xlabel('Amount')
                ax2.set_title('AMT_INCOME_TOTAL')
                ax2.get_xaxis().get_major_formatter().set_scientific(False)
                ax2.get_yaxis().get_major_formatter().set_scientific(False)
                plt.axvline(amt_income_client, color='red')
                
                pl.tight_layout()
                st.pyplot(pl) 
                
                st.write("Les courbes rouges représentent vos caractéristiques.")
                

if choix=='Consulter la base de données':
    st.write('Vous souhaitez consulter la base de données.')

    choix_database = st.selectbox("Quelle base souhaitez-vous consulter ?",
                         ('','Base de données globale.',
                          'Comment est accordé un crédit ?', 
                          "Quelles sont les conditions financières des clients ?",
                          "Quelle tranche d'âge rembourse le mieux son crédit ?"))
    
    # permet de naviguer dans la bdd :
    if choix_database=='Base de données globale.':
        st.write(app_train[features_affichage_2])
    
    # features importances du modèle de prédiction :
    if choix_database=="Comment est accordé un crédit ?":

        feature_importance_values_modelLGBM = LGBM_best.feature_importances_
        feature_importances_modelLGBM = pd.DataFrame({'feature': features, 'importance': feature_importance_values_modelLGBM})
        
        plot_feature_importances_v2(feature_importances_modelLGBM)
    
    # comparer la situation des clients similaires à un client en particulier, selon ses caractéristiques :    
    if choix_database=='Quelles sont les conditions financières des clients ?':
        
        age = st.number_input('Quel âge avez-vous ?', min_value=18, step=1)  # 👈 this is a widget
        revenu = st.number_input('Quel est votre revenu brut annuel ?', min_value=10000, step=1000)
        
        tranche_age = app_train[app_train['DAYS_BIRTH']>(age+2)*(-365)]
        tranche_age2 = tranche_age[tranche_age['DAYS_BIRTH']<(age-2)*(-365)]
        
        tranche_revenu = tranche_age2[tranche_age2['AMT_ANNUITY']>revenu-2000]
        tranche_revenu2 = tranche_revenu[tranche_revenu['AMT_ANNUITY']<revenu+2000]
        
        st.write(tranche_revenu2.shape[0], 'client(s) dans votre catégorie.')
        
        df_finance = tranche_revenu2[['AMT_CREDIT','AMT_INCOME_TOTAL',
                                      'AMT_GOODS_PRICE','AMT_ANNUITY']]
        st.write(df_finance)              
        
        # moyenne des clients dans la même catégorie :
        moy_credit = df_finance['AMT_CREDIT'].mean()
        moy_income = df_finance['AMT_INCOME_TOTAL'].mean()
        moy_goods = df_finance['AMT_GOODS_PRICE'].mean()
        
        #affichage des chiffres :
        if tranche_revenu2.shape[0]!=0:        
            # Affichage des graphes :
            pl.subplot(3, 1, 1)                
            ax1 = sns.distplot(app_train['AMT_CREDIT'])
            ax1.set_xlabel('Amount')
            ax1.set_title('AMT_CREDIT')
            ax1.get_xaxis().get_major_formatter().set_scientific(False)
            ax1.get_yaxis().get_major_formatter().set_scientific(False)
            plt.axvline(moy_credit, color='red')

            pl.subplot(3, 1, 2)
            ax2 = sns.distplot(app_train['AMT_INCOME_TOTAL'])
            ax2.set_xlabel('Amount')
            ax2.set_title('AMT_INCOME_TOTAL')
            ax2.get_xaxis().get_major_formatter().set_scientific(False)
            ax2.get_yaxis().get_major_formatter().set_scientific(False)
            plt.axvline(moy_income, color='red')
        
            pl.subplot(3, 1, 3)
            ax3 = sns.distplot(app_train['AMT_GOODS_PRICE'])
            ax3.set_xlabel('Amount')
            ax3.set_title('AMT_GOODS_PRICE')
            ax3.get_xaxis().get_major_formatter().set_scientific(False)
            ax3.get_yaxis().get_major_formatter().set_scientific(False)
            plt.axvline(moy_goods, color='red')
                
            pl.tight_layout()
            st.pyplot(pl) 
        
            st.write("Les courbes rouges verticales sont à la moyenne des clients dans votre catégorie.")


    if choix_database=="Quelle tranche d'âge rembourse le mieux son crédit ?" :
        
        plt.figure(figsize = (6, 4))

        # KDE plot of loans that were repaid on time
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / (-365), label = 'target == 0')

        # KDE plot of loans which were not repaid on time
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / (-365), label = 'target == 1')

        # Labeling of plot
        plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages')
        plt.legend(loc='upper right')
        st.pyplot()
 
## 