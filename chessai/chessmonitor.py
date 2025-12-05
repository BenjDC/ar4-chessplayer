from stockfish import Stockfish
import chess
import chess.svg
import tkinter as tk
from tkinterhtml import HtmlFrame


class Chessfish:

    def  __init__(self):
        # Chemin vers le binaire Stockfish
        #stockfish_path = "/Users/Myriametben/Documents/GitHub/ar4-chessplayer/stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows
        #stockfish_path = "./stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows
        stockfish_path = "C:\\Users\\Robotique\\Documents\\Benjamin\\ar4-chessplayer\\ar4-chessplayer\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"

        #initialiser affichage 
        # self.root = tk.Tk()
        # self.root.title("Position en cours")

        # Initialiser Stockfish
        self.ai_player = Stockfish(stockfish_path, depth=15)  # depth peut être ajusté

        # Position connue (FEN de départ)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        # Initialiser l'échiquier
        self.board = chess.Board(fen)

        # self.frame = HtmlFrame(self.root, horizontal_scrollbar="auto")
        # self.frame.pack(fill="both", expand=True)

        #self.display_board()

    def display_board(self):
        svg_code = chess.svg.board(self.board)
        self.frame.set_content(svg_code)

    def ai_plays_move(self):

        self.ai_player.set_fen_position(self.board.fen()) 

        # Laisser Stockfish calculer le coup noir
        coup_noir = self.ai_player.get_best_move()
        if coup_noir is None:
            print("Fin de partie !")
            return 
        move = chess.Move.from_uci(coup_noir)

        self.board.push(move)

        print("Stockfish joue :", coup_noir)
        print(self.board)

        #self.display_board()

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
        #self.display_board()
    
    def get_fen(self):
        return self.board.fen()
    

    def endgame(self):
        print(self.board.fen())
        print (self.board)
        self.display_board()
        return