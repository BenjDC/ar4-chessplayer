# 007 – Vision : ArUco et classification des cases par ResNet18

- **Statut** : décidée (en place)
- **Date** : 2025-2026
- **Exigences liées** : EX-02, EX-20, EX-21

## Décision
- Localisation du plateau par 4 tags ArUco `DICT_4X4_50` de 3 cm aux coins (id 19, 29, 39 et 9), puis redressement en 800 × 800 px.
- Découpage en 64 cases, chacune classée par un ResNet18 affiné (PyTorch) en `vide`, `piece_blanche` ou `piece_noire`.
- Le coup est déduit en comparant l'occupation avec la position connue (FEN) : ce n'est pas la reconnaissance du type de pièce qui le donne.
- Les essais Keras et TensorFlow sont abandonnés au profit de PyTorch.

## Conséquences
- Il n'est pas nécessaire de reconnaître le type de pièce : la promotion est supposée en dame, sauf indication contraire.
- Dette : la taille des images à l'entraînement (128 px) diffère de celle en jeu (100 px). Premiers essais sur le portique de test : environ 90 % de reconnaissance, insuffisant pour jouer sans aide (voir 012).
