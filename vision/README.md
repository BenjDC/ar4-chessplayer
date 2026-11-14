# Vision – détection du plateau et du coup joué

Retour au [README principal](../README.md) · Vue d'ensemble du logiciel : [docs/logiciel.md](../docs/logiciel.md)

> Statut : prototype fonctionnel, environ 90 % de reconnaissance. La façon de détecter le coup reste une décision ouverte (vision, capteurs Hall ou RFID) : voir la [décision 012](../docs/decisions/012-detection-du-coup-humain.md).

## Chaîne de traitement – `vision.py`

| Étape | Fonction | Détail |
|---|---|---|
| Capture | `get_board()` | Webcam (`VideoCapture(0)`) ou image de test |
| Localisation | `detect_aruco()`, `get_board_corners_from_tags()` | 4 tags ArUco `DICT_4X4_50` de 3 cm aux coins : id 19 (sous A1), 29 (sous H1), 39 (à droite de H8), 9 (à gauche de A8). Images à imprimer dans `ressources/` |
| Redressement | `warp_board()` | Transformation perspective vers une image 800 × 800 px |
| Découpage | `slice_into_64_cases()` | 64 vignettes de 100 × 100 px, indexées `a1`…`h8` |
| Classification | `predict_board_occupancy()` | ResNet18 (PyTorch, `model_chess.pth`), 3 classes : `piece_blanche`, `piece_noire`, `vide` |
| Déduction du coup | `detect_move_from_occupancy()` | Compare l'occupation avant (issue de la FEN) et après ; gère les roques, les prises et la promotion (dame par défaut) |

Le type des pièces n'est pas reconnu : le coup se déduit des changements d'occupation, connaissant la position précédente.

## Entraînement et données

- `../get_dataset.py` : capture automatique d'images étiquetées, à partir de la position initiale.
- `dataset.zip` : jeu de données d'entraînement.
- `train_pytorch_model.py` : affinage d'un ResNet18 pré-entraîné sur ImageNet (images de 128 px, augmentation, arrêt anticipé), qui produit `model_chess.pth`.
- `arucotest.py` : test de détection des tags.

**Historique (ne servent plus)** : `dataset_train.py`, `dataset_train_v2.py` et les modèles `model_occupation_*.keras`. Ce sont les premières versions sous TensorFlow/Keras (CNN maison, puis apprentissage par transfert).

## Banc de test

Webcam fixée sur un portique en profilés 20 × 20, à la verticale de l'échiquier posé sur la table (voir [décision 011](../docs/decisions/011-montage-camera.md)).

## Dette connue

- Le modèle est entraîné sur des images de 128 px mais reçoit en jeu des cases de 100 px, sans redimensionnement.
- `get_board()` peut renvoyer `(None, None)` au lieu de la liste des cases.
- Le taux de 90 % reste à mesurer par **coup** plutôt que par case (EX-20).
