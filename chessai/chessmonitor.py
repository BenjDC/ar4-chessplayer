from stockfish import Stockfish
import chess


def init_chess():
    # Chemin vers le binaire Stockfish
    #stockfish_path = "/Users/Myriametben/Documents/GitHub/ar4-chessplayer/stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows
    stockfish_path = "./stockfish/stockfish-macos-m1-apple-silicon"  # ou "C:/chemin/vers/stockfish.exe" sous Windows

    # Initialiser Stockfish
    ai_player = Stockfish(stockfish_path, depth=15)  # depth peut être ajusté

    # Position connue (FEN de départ)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Initialiser l'échiquier
    board = chess.Board(fen)

    return board, ai_player


def ai_plays_move(board, ai_player):

    ai_player.set_fen_position(board.fen()) 

    # Laisser Stockfish calculer le coup noir
    coup_noir = ai_player.get_best_move()
    if coup_noir is None:
        print("Fin de partie !")
        return 
    move = chess.Move.from_uci(coup_noir)

    board.push(move)

    print("Stockfish joue :", coup_noir)
    print(board)

    return board


def human_plays_move(human_move, board):

    # Convertir la chaîne en objet coup
    try:
        move = chess.Move.from_uci(human_move)
    except ValueError:
        print(f"Coup {human_move} au format UCI invalide")
        return None

    # Vérifier si le coup est légal
    if move and move in board.legal_moves:
        print(f"Tu joues {human_move}")
    else:
        print(f"Le coup {human_move} est illégal dans cette position.")
        return None

    board.push(move)
    
    return board

def endgame(board):
    print(board.fen())
    print (board)
    return