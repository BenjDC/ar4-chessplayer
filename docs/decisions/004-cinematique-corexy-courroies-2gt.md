# 004 – Cinématique CoreXY, courroies 2GT de 9 mm

- **Statut** : décidée
- **Date** : 2026
- **Exigences liées** : EX-14, EX-15, EX-16

## Décision
- CoreXY classique à deux courroies indépendantes. Les moteurs sont fixes à l'arrière du caisson.
- 2 moteurs NEMA 17 Motech MT-1704HS168A (1,8°/pas, 1,68 A, axe de 5 mm).
- Courroies 2GT renforcées fibre de verre, largeur 9 mm (Trianglelab).
- 10 poulies : 2 motrices de 20 dents (alésage 5 mm) et 8 folles de 9-10 mm, par paires dans les coins avant et sur les chariots Y.
- Croisements : deux étages verticaux décalés d'environ 11-12 mm. Supports de poulies en sandwich, tenus en haut et en bas.

## Conséquences
- Résolution théorique : 2 mm × 20 dents / 200 pas = 0,2 mm par pas complet, soit 0,0125 mm en 1/16 de micropas. C'est largement suffisant pour EX-14.
- Le modèle Onshape utilise pour l'instant un NEMA 17 générique (17HD40). Il faut vérifier la longueur du corps du Motech.
- Les poulies, les courroies et leurs supports ne sont pas encore modélisés.
