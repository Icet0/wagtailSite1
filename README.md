Wagtail Site
Table des matières
À propos du projet
Structure du projet
Déploiement
Dépendances
Utilisation
Contribuer
Licence
Contact
À propos du projet
Ce projet est une application web basée sur Wagtail, un CMS construit sur Django. Il a été développé dans le cadre d'un stage et contient plusieurs fonctionnalités et modules.

Structure du projet Wagtail

Structure du projet
Le projet est organisé en plusieurs applications Django et modules :

Applications Django
architecture
blog
context
dashboard
features
home
loadingData
opinion
prediction
search
visualisation
workflow
Utilitaires
myUtils
Fichiers médias et statiques
media
static
Templates
templates
Déploiement
Le projet utilise Docker pour le déploiement. Les fichiers nécessaires pour le déploiement sont :

Dockerfile
docker-compose.yml
Étapes de déploiement
Construisez l'image Docker en utilisant la commande suivante :
bash
Copy code
docker build -t wagtail-site .
Démarrez les services avec Docker Compose :
bash
Copy code
docker-compose up
Accédez à l'application via votre navigateur web.
Dépendances
Les dépendances du projet sont listées dans les fichiers suivants :

requirements.txt
requirements_mac.txt
Pour installer les dépendances, exécutez :

bash
Copy code
pip install -r requirements.txt
Utilisation
Après le déploiement, vous pouvez accéder à l'application via votre navigateur web et utiliser les différentes fonctionnalités disponibles.

Contribuer
Si vous souhaitez contribuer au projet, veuillez suivre les directives de contribution.

Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

Contact
Si vous avez des questions ou des suggestions, n'hésitez pas à ouvrir une issue ou à me contacter directement.
