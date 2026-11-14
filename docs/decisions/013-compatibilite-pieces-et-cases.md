# 013 – Compatibilité entre la taille des pièces et celle des cases

- **Statut** : ouverte
- **Exigences liées** : EX-11, EX-14, EX-30
- **Risque lié** : R-02
- **Données** : [mesures des pièces](../mesures/pieces.md)

## Contexte
Les bases des pièces Damier Opéra de taille 5 font de 27,5 mm (pion) à 38,8 mm (roi), sur des cases de 50 mm. Les mesures montrent qu'**aucune pièce ne peut longer une ligne entre deux cases occupées** sans toucher une voisine. Le passage par un coin laisse au mieux 0,5 à 8 mm de marge, et provoque un contact dès qu'un roi ou une dame est concerné. Même le cavalier qui sort de sa position initiale frôle les pions.

EX-30 (garder les pièces du commerce) et EX-11 (ne déplacer aucune autre pièce) sont donc en tension.

## Options à étudier
1. **Accepter les contacts légers.** Les pièces lestées glissent sans se renverser ; on vérifie après coup que la voisine est restée dans sa case, au besoin avec un recentrage par l'aimant. C'est la solution la plus simple, compatible avec la démarche V1. Il faut la tester au banc.
2. **Écarter temporairement la pièce gênante**, puis la remettre en place. Pas de contact, mais plus de mouvements et un planificateur plus complexe.
3. **Imprimer des pièces maison** sur la Bambu Lab, avec des bases plus petites (environ 25 mm au plus) et un logement pour un aimant ou un lest en acier. Cela résout en même temps la question des pièces plombées non magnétiques (009). En contrepartie, on s'écarte d'EX-30, et l'esthétique et la stabilité seront à soigner.
4. **Des pièces plus petites** du commerce (taille 4), ou des cases plus grandes (taille 6, 57 mm). Mais la taille 6 donne un échiquier de 456 mm, ce qui réduit la marge dans le caisson de 600 mm.

## Démarche proposée
- **V1** : option 1, avec intervention humaine pour réaligner une pièce si besoin.
- **En parallèle** : une simulation sur de vraies parties, pour mesurer la fréquence des coups problématiques avec les options 1 et 2. Et un essai d'impression d'une ou deux pièces maison (option 3) lors du banc magnétique.
