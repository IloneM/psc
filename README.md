# PSC : "transcription automatique de partitions à partir d'une analyse sonore"

## Utilisation de Git

### Instructions d'initialisation

1.  Créer un compte [github](https://github.com/)
2. 	Installer [git](http://git-scm.com/)
3.  Installer [TortoiseGit](https://tortoisegit.org/download/)
4.  Une fois TortoiseGit installé, ouvrir file explorer et se rendre dans un répertoire dédié au PSC (à créer le cas échéant..) puis cliquer droit à un endroit vide
5.  Se rendre dans le menu "TortoiseGit" puis "Settings"
6.  Se rendre dans le menu "Git" sans faire attention aux éventuelles popups puis séléectionner "global" et entrer un nom et un email (de préférence ceux utilisés pour github)
7.  Se rendre dans le menu "Network" et cocher "Enable Proxy Server" puis entrer:
  + "Server adress": **kuzh.polytechnique.fr**
  + "Port": **8080**
8. Cliquer sur "Appliquer" puis "Ok"
9. Cliquer droit comme précédemment et se rendre sur le menu "Git Clone"
10. Entrer:
  + "URL": https://github.com/IloneM/psc.git
  + "Directory": *le dossier de travail choisi*
11. Cliquer sur OK
12. Si tout se passe bien, le message "Success (*time in ms* @ *timecode*)" devrait apparaître en bleu. Cliquer alors sur "Close". Se rendre dans le dossier: les fichiers du psc devraient s'y trouver.

### Instructions d'utilisations

#### Recevoir les modifications des autres (pull)

1.  Se rendre dans le dossier de travail via File Explorer
2.  Cliquer droit à un endroit vide puis sélectionner 'Git Sync...'
3.  Cliquer sur "Pull". Un message "master|origin/master|origin/HEAD" devrait apparaître sur le fenêtre. La synchronisation est alors terminée
4.  

#### Envoyer ses propres modifications (push)

1.  Se rendre dans le dossier de travail via File Explorer
2.  Cliquer droit à un endroit vide puis sélectionner 'Git Commit -> "master"...'
3.  Entrer un message résumant les modifications apportées
4.  Si des fichiers ont été créées, les séléectionner dans la section "Changes made"
5.  Appuyer sur "Commit"
6.  Une nouvelle fenêtre s'ouvre où apparaît un message au dessous duquel se trouve le bouton "Push". Appuyer sur ce dernier.
7.  Une nouvelle fenêtre s'ouvre demandant les identifiants Github. Les entrer. Si tout est en ordre, un message "Success (*time in ms* @ *timecode*)" devrait apparaître signifiant que les modifications ont été apportées sur le serveur.

### NB

Git est un outil de versionnage complet, libre et particulièrement efficace. Les procédures simples décrites ci-dessus cache un fonctionnement et des possibilités quelques peu plus complexes, mais qui se révèlent extrêment utiles pour un travail collaboratif voire même individuel. Il permet par exemple de pas perdre ton son travail suite à une mauvaise manip pour peu qu'on se soit imposer une certaine discipline de versionnage. Si vous avez le temps, je vous encourage à lire au moins les premiers chapitre du [Git book](https://git-scm.com/book/en/v2)
