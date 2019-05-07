Pour compiler toutes les versions : make

Pour lancer une execution : 
 - se placer dans un dossier part_*
 - make exec pour lancer une execution en local
 - definir un fichier nommé "hostfile" pour une execution sur plusieurs machines, puis make exec. Le nombre de processus est choisi automatiquement de manière optimale en fonction du nombre de machines. On suppose que les machines utilisées sont quad-core.
 - Pour definir les parametres manuellement, il faut utiliser make exec suivi des variables à définir : make exec VAR=valeur. 
 VAR appartient à {NB_PROC, SAMPLE, WIDTH, HEIGHT} 

NB_PROC : Nombre de processus
SAMPLE : Nombre de samples
WIDTH, HEIGHT : Dimensions de l'image 
 
