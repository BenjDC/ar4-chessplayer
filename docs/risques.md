# Registre des risques

Les risques qui peuvent faire échouer ou dérailler le projet, avec la façon de les réduire. On le relit à chaque fin de jalon et après chaque test. Les identifiants `R-xx` sont stables : un risque traité passe au statut « fermé », il n'est pas supprimé.

**Cotation**
- **P** : probabilité, de 1 (peu probable) à 3 (probable ou déjà constaté).
- **I** : impact, de 1 (gêne) à 3 (remet en cause la solution).
- **Criticité** = P × I : 6 à 9 est **critique**, 3 à 4 est à surveiller, 1 à 2 est mineur.

Statuts : **ouvert** · **en cours de traitement** · **fermé** (traité ou devenu sans objet)

Dernière revue : 14/11/2026 (revue de conception).

## Synthèse

| ID | Risque | P | I | Crit. | Statut | Liens |
|---|---|---|---|---|---|---|
| R-01 | Force magnétique insuffisante à travers l'entrefer | 3 | 3 | **9** | ouvert | 002, 009, EX-10 |
| R-02 | Collisions entre pièces pendant les déplacements | 3 | 3 | **9** | ouvert (confirmé par les mesures) | EX-11, EX-14, 010, 013 |
| R-03 | Hypothèses non vérifiées issues de l'IA prises pour des faits | 3 | 2 | **6** | en cours de traitement | méthode |
| R-04 | Détection du coup humain trop peu fiable | 3 | 2 | **6** | en cours de traitement | 007, 012, EX-20, EX-21 |
| R-05 | Flèche du plateau : entrefer variable selon la case | 2 | 2 | 4 | ouvert | 003 |
| R-06 | CoreXY : mise en travers, routage sur deux étages laborieux | 2 | 2 | 4 | ouvert | 004 |
| R-07 | Homing sans capteur non répétable | 2 | 2 | 4 | ouvert | 005, EX-33 |
| R-08 | Perte de pas en cours de partie, plateau désynchronisé | 2 | 2 | 4 | ouvert | 004, 006 |
| R-09 | Aimants dans les pièces qui s'attirent ou se repoussent | 2 | 2 | 4 | ouvert | 009, 012 |
| R-10 | Pas de place accessible pour ranger les pièces prises | 2 | 2 | 4 | ouvert | 010, EX-12 |
| R-11 | Marlin mal adapté à une machine sans chauffe | 2 | 1 | 2 | ouvert | 006 |
| R-12 | Dérive des coûts ou approvisionnement difficile | 2 | 1 | 2 | ouvert | EX-35 |
| R-13 | Chauffe de l'électro-aimant | 1 | 2 | 2 | ouvert | 002, EX-34 |
| R-14 | Quota API Onshape épuisé | 1 | 1 | 1 | ouvert | 008 |
| R-15 | Dépôt alourdi par les modèles binaires | 3 | 1 | 3 | ouvert | dette technique |

## Détail des risques critiques et à surveiller

### R-01 – Force magnétique insuffisante à travers l'entrefer (criticité 9)
- **Cause** : la force d'un aimant chute très vite avec la distance. Les forces annoncées par les fabricants valent au contact d'une pièce d'acier épaisse. Or il y a entre l'aimant et la pièce le plateau (3 à 5 mm), l'échiquier en vinyle, de la feutrine, et une fine rondelle qui sature.
- **Conséquence** : la pièce n'est pas entraînée, ou décroche en cours de trajet.
- **Réduction** : banc magnétique du jalon 2, avant tout achat. On compare l'électro-aimant fixe, l'aimant néodyme levé par un servo, et un aimant dans la pièce, sur 3 matériaux de plateau. Un plateau aussi fin que possible.
- **Indicateur de fermeture** : force latérale de décrochage mesurée au moins 2 fois supérieure à la force de frottement de la pièce la plus lourde.

### R-02 – Collisions entre pièces pendant les déplacements (criticité 9)
- **Cause** : les bases mesurées font de 27,5 mm (pion) à 38,8 mm (roi) ([mesures](mesures/pieces.md)), alors qu'une pièce qui suit la ligne entre deux cases passe à 25 mm du centre des cases voisines. **Confirmé par le calcul** : il y a chevauchement de 2,5 mm (pion contre pion) à 13,8 mm (roi contre roi) dans une gouttière. Par un coin, la marge n'est que de 0,5 à 8 mm, et il y a contact avec un roi ou une dame. Le cavalier qui sort de sa position initiale frôle les pions.
- **Conséquence** : des pièces poussées ou renversées (EX-11), et une position physique qui ne correspond plus à la partie.
- **Réduction** :
  1. Mesurer les bases réelles.
  2. Étude géométrique sur des milliers de parties réelles, pour savoir combien de coups posent problème.
  3. Planificateur de trajets qui contourne les obstacles, ou écarte temporairement une pièce gênante.
  4. Accepter en V1 que l'humain réajuste une pièce.
  5. Envisager des pièces imprimées à bases réduites (décision 013).
- **Indicateur de fermeture** : la stratégie retenue couvre au moins 95 % des coups sans contact, d'après la simulation (seuil à valider).

### R-03 – Hypothèses non vérifiées issues de l'IA prises pour des faits (criticité 6)
- **Cause** : une partie de l'architecture vient d'explorations avec une IA. Elle est cohérente, mais certaines valeurs sont plausibles sans avoir été vérifiées.
- **Réduction** : les décisions sont classées décidée, proposée ou ouverte (`decisions/README.md`). Les grandeurs physiques sont confirmées par un test avant tout achat ou toute conception, et toute valeur chiffrée cite sa source (mesure, fiche technique, calcul).

### R-04 – Détection du coup humain trop peu fiable (criticité 6)
- **Constat** : environ 90 % de reconnaissance avec la vision actuelle.
- **Atténuation existante** : la confirmation humaine en cas de doute (EX-21) rend le risque acceptable en V1.
- **Réduction** : corriger la dette de la vision (taille des images, données d'entraînement, éclairage), mesurer par coup et non par case, et tester un capteur Hall (décision 012).

### R-05 – Flèche du plateau (criticité 4)
- **Cause** : un plateau de 600 mm posé sur le cadre fléchit au centre. D'après une estimation, environ 2 mm pour du PMMA de 5 mm chargé des pièces. L'entrefer, et donc la force, varie selon la case.
- **Réduction** : choisir un matériau rigide (composite aluminium) ou une épaisseur adaptée, et mesurer la flèche réelle avant de fixer la hauteur de l'aimant.

### R-06 – CoreXY : mise en travers et routage sur deux étages (criticité 4)
- **Cause** : des tensions inégales des deux courroies mettent l'axe X en travers. Les deux étages et les supports en sandwich demandent un travail de conception soigné.
- **Réduction** : tendeurs réglables sur chaque boucle, vérification d'équerrage au montage. Plan B : un cartésien simple avec le moteur X embarqué, puisque la vitesse est secondaire.

### R-07 – Homing sans capteur non répétable (criticité 4)
- **Réduction** : prévoir dès la conception l'emplacement de deux micro-switchs de secours, et mesurer la répétabilité de l'origine sur 20 cycles.

### R-08 – Perte de pas en cours de partie (criticité 4)
- **Cause** : un frottement ou une collision fait sauter des pas sans que le contrôleur le sache. Les coups suivants sont alors décalés.
- **Réduction** : vitesses et accélérations modérées, prise d'origine régulière (entre deux coups, par exemple), et contrôle par la détection du coup (vision ou capteurs).

### R-09 – Aimants dans les pièces qui interagissent (criticité 4)
- **Cause** : si l'on colle des aimants néodyme sous les pièces (décisions 009 et 012), deux pièces voisines peuvent s'attirer ou se repousser, surtout dans le rangement des prises.
- **Réduction** : petits aimants (Ø 6-10 mm), tous orientés pareil dans un même camp. Test de deux pièces côte à côte sur le banc magnétique.

### R-10 – Pas de place pour ranger les prises (criticité 4)
- **Réduction** : vérifier dans Onshape la zone réellement accessible au chariot hors de l'échiquier. En V1, retrait par l'humain (EX-12).

## Risques mineurs

- **R-11 – Marlin sans chauffe** : désactiver extrudeur, sondes de température et protections thermiques. En cas de blocage, se replier sur une autre configuration ou un autre firmware.
- **R-12 – Coûts et approvisionnement** : privilégier les articles courants (Leroy Merlin pour les plaques, EX-35) et les petits composants standard, et suivre le coût dans la nomenclature.
- **R-13 – Chauffe de l'électro-aimant** : PWM réduit pendant les déplacements, mesure de température pendant le test. Sans objet si l'on retient l'aimant permanent sur servo.
- **R-14 – Quota Onshape** : un export coûte environ 7 appels sur 2 500 par an. Suivre la ligne « Appels API » de `mecanique/onshape/RESUME.md`.
- **R-15 – Modèles binaires dans git** : déplacer `model_chess.pth` vers Git LFS ou une release GitHub, et retirer les anciens `.keras`.
