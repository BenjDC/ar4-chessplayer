# CLAUDE.md – Joueur d'échecs robotisé (CoreXY)

Ce fichier donne le contexte du projet à chaque session d'IA. Il doit rester court : les détails vivent dans `docs/`.

## Le projet

Un échiquier robotisé qui joue contre un humain. Une caméra lit le plateau, Stockfish choisit le coup, et un chariot CoreXY sous le plateau déplace les pièces avec un électro-aimant.

Projet personnel, mené aussi pour apprendre à utiliser l'IA sur tout le cycle : conception, choix techniques, mécanique, électronique, code.

**Démarche : le chemin compte plus que la destination.** On avance par incréments (V1, V2, V3, voir `docs/exigences.md`) : une fonction simple validée d'abord, puis la suivante. L'intervention humaine est acceptée quand elle évite une complexité prématurée (promotion, confirmation d'un coup douteux…). Les performances sont des indicateurs, pas des critères bloquants au début. Proposer des solutions simples et testables avant les solutions complètes.

> Le dépôt s'appelle `ar4-chessplayer` pour des raisons historiques : l'idée initiale était un bras robot AR4. Elle est abandonnée au profit du plateau CoreXY (voir `docs/decisions/001-corexy-plutot-que-bras-ar4.md`). Ne pas réintroduire de code ou de vocabulaire « bras AR4 ».

## Où trouver quoi

| Sujet | Emplacement |
|---|---|
| Exigences (ce que la machine doit faire) | `docs/exigences.md` |
| Décisions techniques, une fiche par choix | `docs/decisions/` (index dans `docs/decisions/README.md`) |
| Nomenclature (pièces, références, statut) | `docs/nomenclature.csv` |
| Plan et jalons | `docs/plan.md` |
| Registre des risques (à relire en fin de jalon et après chaque test) | `docs/risques.md` |
| Vue d'ensemble, librairies | `README.md` (court : renvoie vers les sous-dossiers) |
| Architecture logicielle, boucle de jeu, module robot | `docs/logiciel.md` |
| Vision, moteur d'échecs | `vision/README.md`, `chessai/README.md` |
| Mécanique | `mecanique/README.md` |
| Électronique et firmware (configuration Marlin à venir) | `electronique/README.md` |
| Modèle CAO exporté d'Onshape (lecture seule) | `mecanique/onshape/RESUME.md` et les JSON voisins |
| Anciens fichiers FreeCAD et STEP | `mecanique/` |

## Architecture en bref

- `chessplayer.py` : boucle de jeu (vision → Stockfish → robot).
- `vision/` : ArUco, redressement, ResNet18 à 3 classes sur les 64 cases, déduction du coup (≈ 90 % de reconnaissance). La détection du coup reste une décision ouverte (012) : vision, capteurs Hall ou RFID.
- `chessai/chessmonitor.py` : état de la partie (python-chess) et Stockfish.
- Pilotage du robot : pas encore écrit. Matériel : cadre et rails Y montés, moteurs, courroie et poulies en stock. Cible : un hôte Python qui envoie du G-code à Marlin (CoreXY) sur une SKR Mini E3 V3, par USB série à 115200 bauds, en attendant `ok` après chaque commande.
- Pendule instrumentée (Arduino Nano, port série) : en option.

**Moyens disponibles** : fablab avec imprimante 3D Bambu Lab P2S. Les pièces imprimées sont faites sur place. Le PLA (disponible) suffit pour les prototypes, le banc de test et la plupart des pièces de la V1. Le PETG est recommandé pour les supports de moteurs (chaleur des moteurs) et à envisager pour les supports de poulies si la tension des courroies se relâche (fluage du PLA). Alimentation de labo et outillage courant sur place, le détail est dans l'inventaire (`docs/actions.md`).

**Actions et achats** : `docs/actions.md` (à faire par le porteur du projet) et `docs/courses.md` (liste des courses par lots).

## Conventions

- **Langue** : français pour la doc, les commentaires et les messages. Identifiants de code en anglais ou en français, mais cohérents au sein d'un module.
- **Décisions** : tout choix technique structurant donne une fiche dans `docs/decisions/` (modèle : `docs/decisions/000-modele.md`). Une décision remplacée n'est pas supprimée : son statut passe à « remplacée par NNN ».
- **Nomenclature** : toute pièce ajoutée au design entre dans `docs/nomenclature.csv`.
- **Exigences et risques** : identifiants stables `EX-xx` et `R-xx`. Ne jamais renuméroter. Un risque traité passe à « fermé ».
- **README** : le README racine reste court. Le détail va dans le README du sous-dossier concerné.
- **Unités** : millimètres, degrés, volts, ampères. Coordonnées du plateau en notation d'échecs (`a1`…`h8`), avec a1 côté blancs, à gauche.
- **Git** : une branche et une pull request par sujet. Pas de commit direct sur `main`.

## CAO Onshape

- Document « Caisson » : `did=e18b653be252676ae3b8c36c`, `wid=db663cb86067f77f2d0621fb`.
- **Lecture** : workflow GitHub « Export Onshape », à lancer à la main. Il écrit `mecanique/onshape/`. Ne pas modifier ces fichiers à la main.
- **Écriture** : connecteur MCP FeatureScript d'Onshape Labs (features paramétriques testées dans un workspace séparé).
- **Quota** : 2 500 appels API Onshape par an (cycle renouvelé le 12 avril), partagés entre l'export et le MCP. Prévenir avant toute opération coûteuse.

## Secrets

Jamais de clé dans le dépôt ni dans une conversation. Les clés Onshape sont dans les secrets GitHub `ONSHAPE_API_ACCESS_KEY` et `ONSHAPE_API_SECRET_KEY`. En local, utiliser un fichier `.env`, déjà ignoré par git.

## Commandes utiles

```bash
export STOCKFISH_PATH=/chemin/vers/stockfish  # ou stockfish dans le PATH
python chessplayer.py --color noirs           # partie, la machine joue les noirs
python chessplayer.py --test essai.png        # teste la vision sur une image, sans caméra
python tools/onshape_export.py --did … --wid … # export Onshape (clés dans l'environnement)
```

## Dette technique connue

- Vision : modèle entraîné sur des images de 128 px mais alimenté en cases de 100 px.
- `vision.get_board()` peut renvoyer `(None, None)` au lieu de la liste des cases.
- Modèles binaires lourds suivis par git (`vision/*.keras` jusqu'à 75 Mo, `model_chess.pth` 43 Mo). Les anciens `.keras` ne servent plus.
