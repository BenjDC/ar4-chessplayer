from stockfish import Stockfish
import chess

# Chemin vers le binaire Stockfish
stockfish_path = "/Users/Myriametben/Documents/GitHub/ar4-chessplayer/stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows

# Initialiser Stockfish
stockfish = Stockfish(stockfish_path, depth=15)  # depth peut être ajusté


# Position connue (FEN de départ)
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Initialiser l'échiquier
board = chess.Board(fen)


while True:

    print(board)
    # Lire le coup blanc de l'adversaire
    coup_blanc = input("Entrez le coup blanc (ex: e2e4) : ").strip()

    # Convertir la chaîne en objet coup
    try:
        move = chess.Move.from_uci(coup_blanc)
    except ValueError:
        print("Coup au format UCI invalide")
        move = None

    # Vérifier si le coup est légal
    if move and move in board.legal_moves:
        print(f"Les blancs jouent {coup_blanc}")
    else:
        print(f"Le coup {coup_blanc} est illégal dans cette position.")


    board.push(move) 
    stockfish.set_fen_position(board.fen()) 

    # Laisser Stockfish calculer le coup noir
    coup_noir = stockfish.get_best_move()
    if coup_noir is None:
        print("Fin de partie !")
        break
    move = chess.Move.from_uci(coup_noir)

    
    print("Stockfish joue :", coup_noir)

    board.push(move) 
    stockfish.set_position(board.fen()) 