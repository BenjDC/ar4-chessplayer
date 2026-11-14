# 011 – Montage de la caméra

- **Statut** : ouverte
- **Exigences liées** : EX-20, EX-22
- **Liée à** : 012 (si la détection du coup n'utilise plus la caméra, cette décision devient sans objet)

## Prototype existant
Le banc de test de la vision est un portique en profilés aluminium 20 × 20 (deux montants et une traverse, renforts en MDF). Il tient une webcam à la verticale au-dessus de l'échiquier posé sur la table, avec les 4 tags ArUco aux coins. Il a servi aux premiers essais (environ 90 % de reconnaissance).

## Options à étudier
1. **Mât fixe sur un côté du caisson** : la géométrie est stable, et l'étalonnage ArUco tolère un léger décalage.
2. **Bras déporté repliable** : plus discret au rangement, mais il risque de bouger.
3. **Caméra au plafond ou sur une lampe** : vue parfaitement verticale, mais pas autonome.

## À trancher
Hauteur et angle minimaux pour voir les 4 tags ArUco et les 64 cases sans que les pièces hautes ne se masquent entre elles. Éclairage (éviter les reflets sur le plexiglas).
