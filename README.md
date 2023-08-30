## Table des matières

- [À propos du projet](#à-propos-du-projet)
- [Structure du projet](#structure-du-projet)
- [Déploiement](#déploiement)
- [Dépendances](#dépendances)
- [Utilisation](#utilisation)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Contact](#contact)

## À propos du projet

Ce projet est une application web basée sur Wagtail, un CMS construit sur Django. Il a été développé dans le cadre d'un stage et contient plusieurs fonctionnalités et modules.

![Structure du projet Wagtail](https://showme.redstarplugin.com/d/d:H9LwEzHT)

## Structure du projet

Le projet est organisé en plusieurs applications Django et modules :

### Applications Django

- **architecture**
- **blog**
- **context**
- **dashboard**
- **features**
- **home**
- **loadingData**
- **opinion**
- **prediction**
- **search**
- **visualisation**
- **workflow**

### Utilitaires

- **myUtils**

### Fichiers médias et statiques

- **media**
- **static**

### Templates

- **templates**

## Déploiement

Le projet utilise Docker pour le déploiement. Les fichiers nécessaires pour le déploiement sont :

- **Dockerfile**
- **docker-compose.yml**

### Étapes de déploiement

1. Construisez l'image Docker en utilisant la commande suivante :
    ```bash
    docker build -t wagtail-site .
    ```
2. Démarrez les services avec Docker Compose :
    ```bash
    docker-compose up
    ```
3. Accédez à l'application via votre navigateur web.

## Dépendances

Les dépendances du projet sont listées dans les fichiers suivants :

- **requirements.txt**
- **requirements_mac.txt**

Pour installer les dépendances, exécutez :

```bash
pip install -r requirements.txt
```

## Utilisation

Après le déploiement, vous pouvez accéder à l'application via votre navigateur web et utiliser les différentes fonctionnalités disponibles.

## Contribuer

Si vous souhaitez contribuer au projet, veuillez suivre les directives de contribution.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

Si vous avez des questions ou des suggestions, n'hésitez pas à ouvrir une issue ou à me contacter directement.
