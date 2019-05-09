Pour compiler toutes les versions, se placer à la racine du projet puis `make` <br/>
Pour lancer une exécution : 
- se placer dans un dossier part_* 
- `make exec` pour lancer une exécution en local avec les paramètres par défaut, à savoir : <br/>
	- NB_PROC = 4 
	- SAMPLE = 50
	- WIDTH = 320
	- HEIGHT = 200
	- HOST = hostfile
- La commande `make exec` vérifie automatiquement la présence d’un fichier contenant les machines hosts à disposition. Le nom du fichier doit impérativement être hostfile pour être reconnu et il doit se trouver dans le même répertoire. Si le nom du fichier est différent, alors il faut lancer la commande “make exec” en précisant le nom du fichier hosts, comme indiqué dans le point suivant. Le nombre de processus est choisi automatiquement de manière optimale en fonction du nombre de machines. On suppose que les machines utilisées sont quad-core.

- Pour définir les paramètres manuellement, il faut utiliser `make exec` suivi des variables à définir :`make exec VAR=valeur` avec VAR appartenant à {NB_PROC, SAMPLE, WIDTH, HEIGHT,HOST,MAP}

	- NB_PROC : Nombre de processus
	- SAMPLE : Nombre de samples
	- WIDTH, HEIGHT : Dimensions de l'image 
	- HOST : Pour changer le nom du fichier hostfile par défaut. 
	- MAP : activer l’option “--map-by node” : Procure de meilleurs performances avec les versions multi-threading 

Quelques exemples :
- Exemple pour calculer une image 1920*1080 avec 500 échantillons par pixel : <br/>
`make exec SAMPLE=500 WIDTH=1920 HEIGHT=1080`
- Pour lancer une exécution en répartissant au mieux les processus sur les différents noeuds : <br/>
`make exec MAP="--map-by node"` 




