# PSC : "transcription automatique de partitions à partir d'une analyse sonore"

## Utilisation de Git

### Instructions pour Ubuntu/Linux/OS X (Unix like)

#### Instructions d'initialisation

1.  Créer un compte [github](https://github.com/) et m'envoyer par email vos nom/email utilisés pour ouvrir votre compte github
2.	Si [git](http://git-scm.com/) n'est pas installé (ce qui ne devrait être le cas pour personne), l'installer conformément aux instructions pour [linux](https://git-scm.com/download/linux) ou pour [OS X](https://git-scm.com/download/mac)
3.	Ouvrir un terminal et taper:
```bash
git config --global user.name "YOUR-NAME"
git config --global user.email "YOUR-EMAIL-ADDRESS"
```
**En remplaçant YOUR-NAME et YOUR-EMAIL-ADRESS par ceux utilisés pour github (Cf 1.)**
4. Toujours dans un terminal taper:
```bash
git clone https://github.com/IloneM/psc.git PATH-TO-WDIR
```
**En remplaçant PATH-TO-WDIR par le chemin d'accès à votre dossier de travail pour le code PSC**
Le message suivant devrait apparaître:
> Unpacking objects: 100% (*un nombre*/*un nombre*), done.

Si tel n'est pas le cas, cela peut-être dû à un problème de proxy:
Si vous êtes à l'X:
	1. Vérifier que le proxy est bien configuré en regardant par exemple si [](https://google.com) est accessible
	2. Si le proxy est bien configuré, essayer de taper dans un terminal:
```bash
git config --global http.proxy http://kuzh.polytechnique.fr:8080
```
Si vous n'êtes pas à l'X:
	1. Vérifier que le proxy est bien désactivé en regardant par exemple si [](https://google.com) est accessible
	2. Si le proxy est bien configuré, essayer de taper dans un terminal:
```bash
git config --global --unset http.proxy
```
**Si aucune de ces solutions ne marche, me contacter svp.**
5. Enfin, toujours dans un terminal taper:
```bash
git config --global credential.helper cache
```

#### Instructions d'utilisations

##### Recevoir les modifications des autres (pull)

1.  Se rendre dans le dossier de travail via la commande [cd](http://linuxcommand.org/lc3_man_pages/cdh.html)
2.  Taper dans un terminal:
```bash
git pull
```
Un des messages suivants devrait apparaître:
*	> Already up-to-date.
Ce qui signifie qu'aucun changement n'a été apporté au code
*	> Unpacking objects: 100% (*n*/*n*), done.
	> From https://github.com/IloneM/psc
	>	*hash code*..*another hash code*  master     -> origin/master
	> Updating *hash code*..*another hash code*

Ce qui signifie que les modification du code ont bien été téléchargées

Si des difficultés sont rencontrées, se reporter à la fin de la section *Instructions d'initialisation*

##### Envoyer ses propres modifications (push)

**Ne fonctionne que si je vous ai déclaré comme collaborateur du projet**

1.  Se rendre dans le dossier de travail via la commande [cd](http://linuxcommand.org/lc3_man_pages/cdh.html)
2.	Taper dans un terminal:
```bash
git add .
git commit -a -m "COMMIT-MSG"
git push
```
**Où COMMIT-MSG est un court message décrivant les modifications apportées**
La première fois, git devrait demander un mot de passe et éventuellement un nom d'utilisateur. Dans les deux cas donner ceux utilisés pour le compte github.
3. Si tout se passe bien le message suivant devrait apparaître:
> To https://github.com/IloneM/psc.git
>   *hash code*..*another hash code*  master -> master

Si tel n'est pas le cas:
* Soit vos identifiants sont erronés auquel cas réessayer
* Soit vous ne vous êtes pas déclarés comme contributeurs, auquel cas je vous invite à m'envoyer vos ids par email comme mentionné en début de section
* Soit le proxy est mal configuré, auquel cas relire la fin de la section *Instructions d'initialisation*
* **Une fois toutes ces possibilités écartées et si le problème persiste, me contacter**

### Instructions pour Windows

#### Instructions d'initialisation

1.  Créer un compte [github](https://github.com/) et m'envoyer par email vos nom/email utilisés pour ouvrir votre compte github
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

#### Instructions d'utilisations

##### Recevoir les modifications des autres (pull)

1.  Se rendre dans le dossier de travail via File Explorer
2.  Cliquer droit à un endroit vide puis sélectionner 'Git Sync...'
3.  Cliquer sur "Pull". Un message "master|origin/master|origin/HEAD" devrait apparaître sur le fenêtre. La synchronisation est alors terminée
4.  

##### Envoyer ses propres modifications (push)

**Ne fonctionne que si je vous ai déclaré comme collaborateur du projet**

1.  Se rendre dans le dossier de travail via File Explorer
2.  Cliquer droit à un endroit vide puis sélectionner 'Git Commit -> "master"...'
3.  Entrer un message résumant les modifications apportées
4.  Si des fichiers ont été créées, les séléectionner dans la section "Changes made"
5.  Appuyer sur "Commit"
6.  Une nouvelle fenêtre s'ouvre où apparaît un message au dessous duquel se trouve le bouton "Push". Appuyer sur ce dernier.
7.  Une nouvelle fenêtre s'ouvre demandant les identifiants Github. Les entrer. Si tout est en ordre, un message "Success (*time in ms* @ *timecode*)" devrait apparaître signifiant que les modifications ont été apportées sur le serveur.

### NB

Git est un outil de versionnage complet, libre et particulièrement efficace. Les procédures simples décrites ci-dessus cache un fonctionnement et des possibilités quelques peu plus complexes, mais qui se révèlent extrêment utiles pour un travail collaboratif voire même individuel. Il permet par exemple de pas perdre ton son travail suite à une mauvaise manip pour peu qu'on se soit imposer une certaine discipline de versionnage. Si vous avez le temps, je vous encourage à lire au moins les premiers chapitre du [Git book](https://git-scm.com/book/en/v2)
