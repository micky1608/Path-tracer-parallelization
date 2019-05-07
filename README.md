Pour compiler toutes les versions : make

Pour lancer une execution : 
 - se placer dans un dossier part_*
 - make exec pour lancer une execution en local
 - definir un fichier nommé "hostfile" pour une execution sur plusieurs machines, puis make exec. Le nombre de processus est choisi automatiquement de manière optimale en fonction du nombre de machines. On suppose que les machines utilisées sont quad-core. 
