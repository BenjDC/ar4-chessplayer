# Exigences

Ce que la machine doit faire, indépendamment de la façon de le faire. Les identifiants `EX-xx` sont stables : on n'en renumérote jamais un, on en ajoute.

## Démarche

Le chemin compte autant que la destination. Le projet avance **par incréments** : on valide d'abord une fonction simple, puis on ajoute la suivante.

- **Une intervention humaine est acceptée** quand elle évite une complexité prématurée : promotion, cas particuliers, confirmation d'un coup douteux…
- **Les performances** (durée d'un coup, bruit, taux de reconnaissance) sont des **indicateurs à suivre**, pas des critères bloquants tant que la fonction n'existe pas.
- La colonne **Incrément** indique quand une exigence doit être tenue :
  - **V1** : premier fonctionnement de bout en bout, avec de l'aide humaine si besoin.
  - **V2** : autonomie sur les coups courants.
  - **V3** : finition (cas rares, confort, performances).

Statuts : **validée** · **à valider** · **abandonnée**

Toutes les exigences ont été validées le 14/11/2026. Les valeurs chiffrées restent des **indicateurs** ou des objectifs par incrément, pas des critères bloquants.

## Jeu

| ID | Exigence | Incrément | Statut |
|---|---|---|---|
| EX-01 | La machine joue une partie complète contre un humain, d'un côté ou de l'autre (blancs ou noirs, au choix au lancement). | V1 | validée |
| EX-02 | Elle connaît le coup joué par l'humain. Le moyen reste ouvert (vision, plateau instrumenté…, voir décision 012). En V1, une confirmation ou une correction humaine est acceptée. Objectif : aucune saisie manuelle. | V1 → V3 | validée |
| EX-03 | Elle choisit ses coups avec Stockfish, avec une force réglable. | V1 (force fixe), V2 (réglable) | validée |
| EX-04 | Elle refuse un coup humain illégal et le signale, sans corrompre l'état de la partie. | V1 | validée |
| EX-05 | Elle connaît tous les coups spéciaux (prise, roque, prise en passant, promotion) dans son suivi de partie. | V1 | validée |
| EX-06 | Elle détecte la fin de partie (mat, pat, nulle) et l'annonce. | V2 | validée |

## Déplacement des pièces

| ID | Exigence | Incrément | Statut |
|---|---|---|---|
| EX-10 | Elle déplace ses propres pièces sur le plateau physique, sans intervention humaine. | V1 (coups simples) | validée |
| EX-11 | Un déplacement ne renverse ni ne décale aucune autre pièce. | V1 | validée |
| EX-12 | Elle retire du plateau les pièces qu'elle capture et les range hors de la zone de jeu. En V1, le retrait peut être fait par l'humain sur demande. | V1 aidée, V2 autonome | validée (emplacement : décision 010) |
| EX-13 | Elle exécute le roque et la prise en passant. Pour la promotion, l'humain place la pièce en V1 et en V2. | V2 (roque, en passant), V3 (promotion) | validée |
| EX-14 | Précision de positionnement de la pièce, c'est-à-dire de la chaîne mécanique **et** du couplage aimant-pièce. Les passages par les coins laissent 0,5 à 8 mm de marge ([mesures](mesures/pieces.md)) : ±3 mm est trop large. Objectif provisoire : environ ±1 mm, à confirmer au banc (décalage entre la position d'arrêt de l'aimant et celle de la pièce). | V1 (fonctionnel), V2 (±1 mm) | validée |
| EX-15 | Durée d'exécution d'un coup. Indicateur suivi, sans objectif en V1. Repères pour la V3 : ≤ 15 s pour un coup simple, ≤ 30 s pour une prise. | V3 | validée |
| EX-16 | Fonctionnement assez silencieux pour une pièce de vie. | V3 | validée |

## Détection du coup humain

| ID | Exigence | Incrément | Statut |
|---|---|---|---|
| EX-20 | Taux de reconnaissance correcte du coup humain. Mesure actuelle de la vision : environ 90 %. Indicateur à suivre, objectif de la V3 : ≥ 99 %. | V3 | validée |
| EX-21 | En cas de doute (coup ambigu ou illégal d'après la détection), la machine demande une confirmation plutôt que de jouer sur une erreur. | V1 | validée |
| EX-22 | Le dispositif de détection (caméra et son support, ou capteurs) ne gêne ni le joueur ni la vue du plateau. | V2 | validée (voir décisions 011 et 012) |

## Matériel et usage

| ID | Exigence | Incrément | Statut |
|---|---|---|---|
| EX-30 | Échiquier taille 5 (cases de 50 mm) et pièces du commerce (Damier Opéra), modifiées au minimum. | V1 | validée |
| EX-31 | Encombrement : caisson de 600 × 600 × 100 mm au plus, posable sur une table. | V1 | validée |
| EX-32 | Une seule alimentation secteur (bloc 24 V). | V1 | validée |
| EX-33 | Mise en route simple : allumer, lancer le programme, prise d'origine automatique, jouer. | V2 | validée |
| EX-35 | Maîtrise des coûts : matériaux et fournitures bon marché et courants. Les éléments volumineux, difficiles à envoyer par la poste (plaques, panneaux), sont pris dans un magasin Leroy Merlin proche, avec découpe en magasin si possible. | V1 | validée |
| EX-34 | Sécurité : pas de pièce mobile ni de tension accessible au joueur en usage normal. L'électro-aimant ne doit pas surchauffer (PWM réduit pendant les déplacements). | V1 | validée |

## En option

| ID | Exigence | Incrément | Statut |
|---|---|---|---|
| EX-40 | Pendule instrumentée : l'appui de l'humain sur la pendule déclenche la détection du coup. | V2 | validée (en option) |
| EX-41 | La machine appuie elle-même sur la pendule après son coup. | V3 | validée |
