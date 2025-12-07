from stockfish import Stockfish
import chess
import chess.svg
import tkinter as tk
from PIL import Image, ImageTk
import io


class Chessfish:

    def  __init__(self):
        stockfish_path = "/Users/Myriametben/Documents/GitHub/ar4-chessplayer/stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows
        #stockfish_path = "C:\\Users\\Robotique\\Documents\\Benjamin\\ar4-chessplayer\\ar4-chessplayer\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"

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
            svg_kwargs['arrows'] = [chess.svg.Arrow(from_sq, to_sq, color="#008800")]
        
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
        
    def ai_plays_move(self):

        self.ai_player.set_fen_position(self.board.fen()) 

        # Laisser Stockfish calculer le coup noir
        coup_noir = self.ai_player.get_best_move()
        if coup_noir is None:
            print("Fin de partie !")
            return 
        move = chess.Move.from_uci(coup_noir)

        # Afficher d'abord la flèche du coup
        self.update_display(arrow=(move.from_square, move.to_square))

        # Attendre un input pour confirmer le coup
        input("Merci de jouer mon coup !")

        self.board.push(move)

        

        print("Stockfish joue :", coup_noir)
        print(self.board)

        self.update_display()

    def human_plays_move(self, human_move):
        print('je joue le coup ' + str(human_move))

        # Convertir la chaîne en objet coup
        try:
            move = chess.Move.from_uci(human_move)
        except ValueError:
            print(f"Coup {human_move} au format UCI invalide")
            return

        # Vérifier si le coup est légal
        if move and move in self.board.legal_moves:
            print(f"Tu joues {human_move}")
        else:
            print(f"Le coup {human_move} est illégal dans cette position.")
            return

        self.board.push(move)
        self.update_display()
    
    def get_fen(self):
        return self.board.fen()
    