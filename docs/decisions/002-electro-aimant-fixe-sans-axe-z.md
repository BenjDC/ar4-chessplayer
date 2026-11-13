# 002 – Électro-aimant fixe, sans axe Z

- **Statut** : décidée
- **Date** : 2026
- **Exigences liées** : EX-10, EX-11, EX-34

## Contexte
Il faut saisir puis relâcher une pièce à travers le plateau.

## Options étudiées
1. **Aimant permanent monté sur un axe Z** (servo ou moteur) : aucune consommation au maintien. Mais il ajoute de la masse et de la complexité sur le chariot, et le relâchement dépend de la course.
2. **Électro-aimant fixe sur le chariot** : commandé électriquement, sans pièce mobile supplémentaire, avec une force réglable en PWM.

## Décision
Électro-aimant 24 V, de 25 à 50 N de maintien, Ø 20-25 mm, monté fixe sur le chariot central.

## Conséquences
- Commande PWM : 100 % pour saisir, environ 50 % pendant le déplacement, pour limiter la chauffe et l'attraction des pièces voisines.
- Magnétisme résiduel : feutrine ou téflon sur l'aimant, et réglage précis de sa hauteur sous le plexiglas.
- Sortie MOSFET de la carte et diode de roue libre (voir 005).
- À valider par le test magnétique du jalon 2.
