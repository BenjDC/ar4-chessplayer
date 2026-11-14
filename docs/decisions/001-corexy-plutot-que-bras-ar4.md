# 001 – Plateau CoreXY plutôt que bras robot AR4

- **Statut** : décidée
- **Date** : 2026 (avant la reprise du dépôt)
- **Exigences liées** : EX-10, EX-11, EX-16, EX-31, EX-34

## Contexte
Le projet est né autour d'un bras robot AR4 (6 axes) qui saisirait les pièces par-dessus le plateau. D'où le nom du dépôt.

## Options étudiées
1. **Bras AR4 au-dessus du plateau** : spectaculaire, saisie directe de n'importe quelle pièce. Mais il est coûteux, encombrant et bruyant, demande une calibration délicate, risque de heurter le joueur et masque le plateau à la caméra.
2. **Portique cartésien sous le plateau avec électro-aimant** : invisible, compact et silencieux, et la caméra garde une vue dégagée. En contrepartie, les pièces glissent au lieu d'être soulevées, et il faut éviter les collisions.
3. **CoreXY sous le plateau avec électro-aimant** : comme l'option 2, mais les deux moteurs restent fixes, ce qui donne une partie mobile légère.

## Décision
Option 3 : plateau CoreXY dans un caisson sous l'échiquier, avec un électro-aimant qui fait glisser les pièces.

## Conséquences
- Les pièces doivent être attirables par un aimant (voir décision 009).
- Les trajectoires passent par les lignes entre les cases pour éviter les collisions (EX-11).
- Le code et la doc ne parlent plus de « bras » mais du « robot » ou du « CoreXY ».
