from stockfish import Stockfish
import chess
import chess.svg
import tkinter as tk
from PIL import Image, ImageTk
import io
import platform
import os
import shutil


class Chessfish:

    def  __init__(self):

        # Chemin de l'exécutable Stockfish : variable d'environnement STOCKFISH_PATH,
        # sinon recherche de « stockfish » dans le PATH.
        stockfish_path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
        if not stockfish_path:
            raise RuntimeError(
                "Exécutable Stockfish introuvable : définir la variable d'environnement "
                "STOCKFISH_PATH (chemin complet de l'exécutable) ou ajouter stockfish au PATH."
            )

        # Initialiser Stockfish
        self.ai_player = Stockfish(stockfish_path, depth=15)  # depth peut être ajusté

        # Position connue (FEN de départ)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Initialiser l'échiquier
        self.board = chess.Board(fen)

        # Fenêtre Tkinter pour affichage
        self.root = tk.Tk()
        self.root.title("Chessfish")
        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        # Affichage initial
        self.update_display()
        self.root.update()

    def update_display(self, arrow=None):
        """Mettre à jour l'affichage avec SVG converti en image Tkinter."""
        """
        arrow : tuple (from_square, to_square) pour afficher une flèche
        """
        svg_kwargs = {}
        if arrow:
            from_sq, to_sq = arrow
            svg_kwargs['arrows'] = [chess.svg.Arrow(from_sq, to_sq, color="#0000F0")]
        
        svg_data = chess.svg.board(self.board, **svg_kwargs)

        # Convertir SVG en PNG via PIL
        try:
            from cairosvg import svg2png
            png_data = svg2png(bytestring=svg_data.encode("utf-8"))
            image = Image.open(io.BytesIO(png_data))
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.config(image=self.tk_image)
        except ImportError:
            # Si cairosvg non installé → affichage ASCII en console
            print("\nÉchiquier ASCII :")
            print(self.board)
            print(f"FEN : {self.board.fen()}")

        self.root.update()

    def get_ai_move(self):
        self.ai_player.set_fen_position(self.board.fen()) 

        # Laisser Stockfish calculer le coup noir
        coup_noir = self.ai_player.get_best_move()
        if coup_noir is None:
            print("Fin de partie !")
            return 
        
        move = chess.Move.from_uci(coup_noir)

        # Afficher d'abord la flèche du coup
        self.update_display(arrow=(move.from_square, move.to_square))

        return move

    def play_move(self, played_move):
        print('coup joué' + str(played_move))

        # Vérifier si le coup est légal
        if not played_move in self.board.legal_moves:
            print(f"Le coup {played_move} est illégal dans cette position.")
            return

        self.board.push(played_move)
        self.update_display()
    
    def get_fen(self):
        return self.board.fen()
    