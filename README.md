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
#### Principes.
_Un GAN se compose de deux réseaux indépendants, un Générateur et un Discriminateur._
_Le Générateur génère des échantillons synthétiques à partir d'un bruit aléatoire (échantillonné à partir d'un espace latent)._
_Le Discriminateur est un classificateur binaire qui distingue si l'échantillon d'entrée est réel (renvoie une valeur scalaire 1) ou faux (renvoie une valeur scalaire 0)._
_Dans la boucle d'entrainement, on construit des vecteurs "tag" qui permettent de codifier les valeurs réelles et les valeurs "fake", issues de l'espace latent, résultant du générateur._

#### 1. Apprentissage discriminator.
_Au début de l'apprentissage, le discriminator reçoit un vecteur de valeurs réelles (codés 1)._
_Puis il reçoit un vecteur issu du générateur constitué de valeurs aléatoires fake (codé 0)._
_La sortie du discriminator est un vecteur de taille [batch_size, 0<= val <= 1]. En effet, juste avant la fonction sigmoid du discriminator on a forcé le nombre de features à 1, puis ce vecteur est transformé par la sgimoid pour donner une valeur comprise entre 0 et 1, valeur qui puisse être comparable aux valeurs de code 0 et 1._
_L'erreur est calculée à partir de la comparaison des deux valeurs, sortie du discriminator et valeur de code._
_Au tout début l'erreur est forcément élevée car l'apprentissage n'a pas encore eu lieu._
_Suite au calcul de l'erreur, par backpropagation, les poids du discriminator sont mis à jour._
_Cette séquence sert au discriminator à apprendre les distributions propres à des valeurs réelles d'une part et des valeurs fake d'autre part._

#### 2. Apprentissage generator.
_Dans un deuxiéme temps, le générator produit un vecteur de valeurs aléatoires et celui ci va être (codé 1)._
_Ce vecteur est présenté au discriminator qui produit une prédiction sur la classification du vecteur présenté (réel ou fake ?)._
_Immédiatement après, on calcule l'erreur en comparant la valeur de prédiction issue du discriminator et la valeur de code, ici 1._
_L'erreur est calculée en compareant la valeur de prédiction et la valeur de code 1._
_Le fait de comparer la prédiction à la valeur 1 pour le calcul de l'erreur revient à dire que l'objectif est que le vecteur généré par le generator soit quasiment identique au vecteur des valeurs réelles._
_L'erreur est forcément élevée au déburt de l'entrainement._
_Par backpropagation, les poids du generateur sont mis à jour._

#### 3. Itération.
_Ce processus se répète au fur et à mesure des epochs._
_Les poids du discriminator et du generator sont de plus en plus précis._
_Au bout d'un moment, les erreurs du discriminator et du genrator sont stabilisées et faibles. La qualité du generator est maximale._

### Consultant @ IBM France.
#### Missions Organisation de Système Information, clients européens.
- Bezecq Telecom
- Orkla Foods
- O2
- Airbus
- Crédit Agricole Markets
- Waberer

