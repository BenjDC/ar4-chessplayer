# 002 – Électro-aimant fixe, sans axe Z

- **Statut** : ouverte. C'est une proposition issue d'une première exploration avec une IA, non consolidée, et aucun matériel n'est acheté. Elle sera tranchée par le banc magnétique du jalon 2.
- **Date** : 2026
- **Exigences liées** : EX-10, EX-11, EX-34

## Contexte
Il faut saisir puis relâcher une pièce à travers le plateau.

## Options étudiées
1. **Aimant permanent monté sur un axe Z** (servo ou moteur) : aucune consommation au maintien. Mais il ajoute de la masse et de la complexité sur le chariot, et le relâchement dépend de la course.
2. **Électro-aimant fixe sur le chariot** : commandé électriquement, sans pièce mobile supplémentaire, avec une force réglable en PWM.

## Options à départager
1. **Électro-aimant fixe** (24 V, Ø 20-25 mm) sous le plateau : pas de pièce mobile. Mais la force annoncée est mesurée au contact : elle chute fortement dès 1 mm d'entrefer, et l'entrefer réel sera de 3 à 7 mm.
2. **Aimant permanent néodyme levé par un micro-servo** : forte attraction sans chauffe ni magnétisme résiduel, la solution la plus courante dans les projets amateurs. La SKR Mini E3 V3 dispose d'un connecteur de sonde qui peut commander un servo (à vérifier).
3. **Électro-aimant et aimant néodyme dans la pièce** : la pièce « active » augmente beaucoup la force à distance.

## Proposition initiale (non retenue à ce stade)
Électro-aimant 24 V, de 25 à 50 N de maintien, Ø 20-25 mm, monté fixe sur le chariot central.

## Conséquences
- Commande PWM : 100 % pour saisir, environ 50 % pendant le déplacement, pour limiter la chauffe et l'attraction des pièces voisines.
- Magnétisme résiduel : feutrine ou téflon sur l'aimant, et réglage précis de sa hauteur sous le plexiglas.
- Sortie MOSFET de la carte et diode de roue libre (voir 005).
- À valider par le test magnétique du jalon 2.
