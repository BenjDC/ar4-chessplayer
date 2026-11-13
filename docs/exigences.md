# Exigences

Ce que la machine doit faire, indépendamment de la façon de le faire. Les identifiants `EX-xx` sont stables : on n'en renumérote jamais un, on en ajoute.

Les valeurs chiffrées marquées **(à valider)** sont des propositions de départ, à confirmer ou à corriger.

Statuts : **validée** · **à valider** · **abandonnée**

## Jeu

| ID | Exigence | Statut |
|---|---|---|
| EX-01 | La machine joue une partie complète contre un humain, d'un côté ou de l'autre (blancs ou noirs, au choix au lancement). | validée |
| EX-02 | Elle reconnaît le coup joué par l'humain à partir d'une photo du plateau, sans saisie manuelle. | validée |
| EX-03 | Elle choisit ses coups avec Stockfish, avec une force réglable. | à valider |
| EX-04 | Elle refuse un coup humain illégal et le signale, sans corrompre l'état de la partie. | validée |
| EX-05 | Elle gère tous les coups spéciaux : prise, roque, prise en passant, promotion. | validée |
| EX-06 | Elle détecte la fin de partie (mat, pat, nulle) et l'annonce. | à valider |

## Déplacement des pièces

| ID | Exigence | Statut |
|---|---|---|
| EX-10 | Elle déplace ses propres pièces sur le plateau physique, sans intervention humaine. | validée |
| EX-11 | Un déplacement ne renverse ni ne décale aucune autre pièce. | validée |
| EX-12 | Elle retire du plateau les pièces qu'elle capture et les range hors de la zone de jeu. | validée (emplacement : décision ouverte 010) |
| EX-13 | Elle exécute le roque (deux pièces) et la promotion. Pour la promotion, une pièce de remplacement doit être disponible, ou l'humain la pose sur demande. | à valider |
| EX-14 | Précision de positionnement : pièce posée à ±3 mm du centre de sa case. **(à valider)** | à valider |
| EX-15 | Durée d'exécution d'un coup simple : ≤ 15 s ; d'une prise : ≤ 30 s. **(à valider)** | à valider |
| EX-16 | Fonctionnement assez silencieux pour une pièce de vie : pas de bruit de moteur gênant à 1 m. **(à valider)** | à valider |

## Vision

| ID | Exigence | Statut |
|---|---|---|
| EX-20 | Classement correct de l'occupation des 64 cases (vide, pièce blanche, pièce noire) dans ≥ 99 % des cas, sous l'éclairage d'une pièce de vie. **(à valider)** | à valider |
| EX-21 | En cas de doute (coup ambigu ou illégal d'après la vision), la machine redemande une photo ou une confirmation, plutôt que de jouer sur une erreur. | à valider |
| EX-22 | La caméra et son support ne gênent ni le joueur ni la vue du plateau. | validée (montage : décision ouverte 011) |

## Matériel et usage

| ID | Exigence | Statut |
|---|---|---|
| EX-30 | Échiquier taille 5 (cases de 50 mm) et pièces du commerce (Damier Opéra), modifiées au minimum. | validée |
| EX-31 | Encombrement : caisson de 600 × 600 × 100 mm au plus, posable sur une table. | validée |
| EX-32 | Une seule alimentation secteur (bloc 24 V). | validée |
| EX-33 | Mise en route simple : allumer, lancer le programme, prise d'origine automatique, jouer. | à valider |
| EX-34 | Sécurité : pas de pièce mobile ni de tension accessible au joueur en usage normal. L'électro-aimant ne doit pas surchauffer (PWM réduit pendant les déplacements). | validée |

## En option

| ID | Exigence | Statut |
|---|---|---|
| EX-40 | Pendule instrumentée : l'appui de l'humain sur la pendule déclenche la lecture du plateau. | validée (en option) |
| EX-41 | La machine appuie elle-même sur la pendule après son coup. | à valider |
