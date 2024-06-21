# PORTFOLIO.
# Data Scientist

## **_Education_**.
- Maîtrise de Sciences et structure de la matière.
- Master in Business Administration.
- Master Ingénierie Financière.
- Master Data Sciences

## **_Experience_**.
### Data scientist (Python/Pytorch/Scikitlearn/Matplotlib/Plotly/Dash...).
#### Projets.
- Volumes de ventes, marque française secteur du luxe (environnement SQL et GCP, plateforme Qlick-autoML): Production POC pour estimation faisabilité (Tests architectures XGBM, LSTM en cours).
- Détection anomalies/prédiction dysfonctionnement groupes électrogènes, activité maintenance industrielle (environnement datalake/SQL sur réseau interne; Cadrage étude (gros vol. d'équipements en maintenance, affectation priorités, compréhension données capteurs), constitution historiques data (séries temporelles), modélisation (VAE, LSTM) et test versus objectifs métiers, développement app. Streamlit et déploiement sur AWS/EC2.
- Automatisation catégories produits, place de marché multi-catégorielle.
- Gestion accord de crédit aux particuliers, activité de crédit consommation.

#### Projets et publications théoriques.
- [VAE](https://github.com/DSAGRO3F/VAE_MNIST)
- [CNN classification](https://github.com/DSAGRO3F/CNN_image_classification)
- [Airflight tickets sales prediction](https://github.com/DSAGRO3F/airflight_pytorch)
- [Predictive maintenance](https://github.com/DSAGRO3F/pm_git_api_V3)
- [Automatisation catégorisation produits place de marché](https://github.com/DSAGRO3F/product_classification_automation)
- [Gestion accord crédit consommation/api](https://github.com/DSAGRO3F/risk_rating_api)
- [Gestion accord crédit consommation/dashboard](https://github.com/DSAGRO3F/risk_rating_dashboard)
- [Analyse prédiction consommation énergétique bâtiment non résidentiel](https://github.com/DSAGRO3F/Analyse_predictive_batiment_energetique_CO2)


## Coup d'oeil sur les modèles génératifs.
### IA générative, l'architecture des VAE.
_L'architecture du VAE se compose d'un encoder et d'un decoder. L'objet de l'encoder est d'extraire les features essentielles pour aboutir à une représentation simple de la donnée d'entrée. Cette représentation simple est nommée l'espace latent._

_Cette représentation de l'espace latent est construite à partir de deux vecteurs: mean et std. Tous deux ont une représentation simple comme on l'a évoqué précédemment. L'une des subtilités du modèle est que le vecteur std est associé à un vecteur de distribution normale centrée par le biais d'un produit scalaire. Le produit de ce produit scalaire est ajouté au vecteur mean pour former un vecteur z._
_Au fur et à mesure des boucles successives, le vecteur de distribution normale prend des valeurs différentes de sorte que le vecteur z prend lui aussi des valeurs différentes._

_Le decoder prend en entrée le vecteur z et les couches nn.Linear(), nn.ReLU() et nn.Dropout() vont transformer z et le faire passer d'un vecteur de dimension simple à un vecteur qui aura, en sortie du decoder, la même dimension que celle du vecteur d'origine. Ce qui permet de tenter de reconstruire la donnée d'entrée avec l'aide de la fonction de perte (qui opère par le processus de descente de gradients)._

_Les données en sortie du decoder sont très semblables à celles en entrée de l'encoder mais jamais identiques à cause de l'architecture du vecteur z dont l'une des composante est une distribution normale._

_Finnalement les poids du modèle, durant leur phase d'entrainement, apprennent la distribution d'un pattern. Ce qui fait des VAE des modèles très intéressants pour la détection d'anomalies dans les secteurs de l'Industrie, de la banque (fraude) par exemple._

- [nuage de représentation des vecteurs de l'espace latent](https://drive.google.com/file/d/103Ic8UWLj6mqW-zEshQiReZxLU3HAItz/view?usp=sharing)
- [decoder output](https://drive.google.com/file/d/1W_-V5tk4TYbYmuquczgfmBYIYuEm4qeo/view?usp=drive_link)


### IA générative, l'architecture des GAN.

### Consultant @ IBM France.
#### Missions Organisation de Système Information, clients européens.
- Bezecq Telecom
- Orkla Foods
- O2
- Airbus
- Crédit Agricole Markets
- Waberer

