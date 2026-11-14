# Joueur d'échecs robotisé (CoreXY)

Un joueur d'échecs robotique. La position est lue sur le plateau (caméra pour l'instant), Stockfish choisit le coup, et un chariot CoreXY sous l'échiquier déplace les pièces avec un aimant.

Le projet avance par incréments (V1, V2, V3) : on valide une fonction simple, puis on ajoute la suivante.

> Le dépôt s'appelle `ar4-chessplayer` pour des raisons historiques : l'idée initiale d'un bras robot AR4 a été abandonnée (voir [décision 001](docs/decisions/001-corexy-plutot-que-bras-ar4.md)).

## Où en est-on ?

- [x] Détection du plateau par ArUco et classification des 64 cases (environ 90 % de réussite)
- [x] Déduction du coup joué et suivi de la partie avec Stockfish
- [x] Cadre MakerBeam assemblé, rails Y posés, moteurs et courroie en stock
- [ ] Banc magnétique : choix du préhenseur et du plateau (jalon 2)
- [ ] Pilotage du CoreXY (firmware et module robot)
- [ ] Pendule instrumentée (en option)

Le détail est dans le [plan](docs/plan.md).

## Organisation de la documentation

| Sujet | Où |
|---|---|
| Logiciel : architecture et boucle de jeu | [docs/logiciel.md](docs/logiciel.md) |
| Vision | [vision/README.md](vision/README.md) |
| Moteur d'échecs | [chessai/README.md](chessai/README.md) |
| Mécanique et CAO | [mecanique/README.md](mecanique/README.md) |
| Électronique et firmware | [electronique/README.md](electronique/README.md) |
| Exigences | [docs/exigences.md](docs/exigences.md) |
| Décisions techniques | [docs/decisions/README.md](docs/decisions/README.md) |
| Registre des risques | [docs/risques.md](docs/risques.md) |
| Nomenclature | [docs/nomenclature.csv](docs/nomenclature.csv) |
| Plan et jalons | [docs/plan.md](docs/plan.md) |
| Actions à mener (inventaire, tests) | [docs/actions.md](docs/actions.md) |
| Liste des courses, par lots | [docs/courses.md](docs/courses.md) |

## Démarrage rapide

```bash
python chessplayer.py --color noirs        # partie : la machine joue les noirs
python chessplayer.py --test essai.png     # teste la vision sur une photo, sans caméra
```

Il faut aussi télécharger l'exécutable Stockfish (https://stockfishchess.org/download/). Son chemin est pour l'instant codé en dur dans `chessai/chessmonitor.py`.

## Librairies utilisées

Python 3. Pour tout installer : `pip install -r requirements.txt`.

> `requirements.txt` est la convention Python pour lister les **dépendances** à installer. Il n'a rien à voir avec les exigences du projet, qui sont dans [docs/exigences.md](docs/exigences.md).

| Librairie (paquet pip) | Utilisée par | Rôle |
|---|---|---|
| `opencv-python` (≥ 4.7, API `cv2.aruco.ArucoDetector`) | `vision/`, `get_dataset.py` | Capture caméra, tags ArUco, redressement, découpage |
| `numpy` | `vision/` | Calcul sur les images |
| `torch`, `torchvision` | `vision/vision.py`, `vision/train_pytorch_model.py` | Modèle ResNet18 de classification des cases |
| `chess` (python-chess) | `chessai/`, `vision/` | Règles, FEN, légalité des coups, rendu SVG |
| `stockfish` + exécutable Stockfish | `chessai/chessmonitor.py` | Choix du coup de la machine |
| `Pillow` | `chessai/chessmonitor.py` | Affichage de l'échiquier dans Tkinter |
| `cairosvg` (+ bibliothèque système Cairo) | `chessai/chessmonitor.py` | Conversion du SVG de l'échiquier en image ; repli en ASCII si absent |
| `tkinter` (bibliothèque standard ; paquet `python3-tk` sous Linux) | `chessai/chessmonitor.py` | Fenêtre d'affichage de la partie |
| `pyserial` | `chessplayer.py` | Port série : pendule, et plus tard la carte de pilotage |
| `requests` | `tools/onshape_export.py` | API REST Onshape (dans GitHub Actions) |
| `tensorflow` | `vision/dataset_train*.py` | **Historique**, inutile pour la chaîne actuelle |

## Matériel de jeu

- Échiquier : [Damier Opéra, vinyle souple roulable taille 5](https://damieropera.com/produit/echiquier-en-vinyle-souple-roulable-4/) (vert)
- Pièces : [Damier Opéra, résine plombée taille 5](https://damieropera.com/produit/pieces-en-resine-plombees-taille-5-3/)
- Pendule : [RNPKZ professionnelle](https://www.amazon.fr/dp/B0FDQJTFJ2/), instrumentée par un Arduino Nano

Le reste du matériel est dans la [nomenclature](docs/nomenclature.csv).
