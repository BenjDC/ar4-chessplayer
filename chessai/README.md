# Moteur d'échecs – `chessmonitor.py`

Retour au [README principal](../README.md) · Vue d'ensemble du logiciel : [docs/logiciel.md](../docs/logiciel.md)

## Classe `Chessfish`

- Tient l'état de la partie avec `python-chess`, à partir de la FEN de départ.
- `get_ai_move()` : meilleur coup selon Stockfish (profondeur 15).
- `play_move()` : vérifie la légalité d'un coup et l'applique.
- Affiche l'échiquier dans une fenêtre Tkinter (SVG de `chess.svg` converti par `cairosvg`). Si Cairo est absent, l'affichage se replie en ASCII dans la console.

## Stockfish

Le moteur Stockfish est un exécutable à télécharger à part (https://stockfishchess.org/download/). Le paquet Python `stockfish` ne fait que le piloter.

**Dette** : les chemins vers l'exécutable sont codés en dur pour deux machines (Windows et macOS) dans `__init__`. À rendre configurable, par variable d'environnement ou argument.
