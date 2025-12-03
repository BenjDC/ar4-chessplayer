import cv2
import sys
import numpy as np
import argparse
from vision import vision
from chessai import chessmonitor


def parse_args():
    parser = argparse.ArgumentParser(
        description="AR4 Chess Player – vision + robot"
    )

    # Paramètre couleur du bras
    parser.add_argument(
        "--color",
        choices=["blancs", "noirs"],
        default="black",
        help="Couleur jouée par le bras (blancs | noirs). Noirs par défaut "
    )

    # Paramètre mode debug vision
    parser.add_argument(
        "--debug-vision",
        action="store_true",
        help="Active les affichages et images intermédiaires."
    )

    # Paramètre pour simuler le robot sans mouvement réel
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne pas envoyer les commandes au bras robotisé."
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("=== AR4 Chess Player ===")
    print(f"Je joue avec les {args.color}")

    board, fishplayer = chessmonitor.init_chess()

    if args.color == "blancs":
        current_player = "ai"
    else:
        current_player = "human"
    
    while(True):
        
        if current_player == "ai":
            board = chessmonitor.ai_plays_move(board, fishplayer)

            if not args.dry_run:
                input("le bras n'est pas prêt, merci de jouer mon coup")
            else:
                input("mode dry run activé, merci de jouer mon coup")

            current_player = "human"
            

        else:
            before_move = vision.get_board()

            if before_move == None:
                print(f"Erreur de vision plateau")
                chessmonitor.endgame(board)
                break

            input("Confirmer lorsque ton coup est joué")

            after_move = vision.get_board()

            state_before, masks_before = vision.board_state(before_move)
                
            #vision.show_occupancy_grid(state_before)

            state_after,  masks_after  = vision.board_state(before_move)

            human_move = vision.detect_move_from_state(state_before, state_after)

            board = chessmonitor.human_plays_move(board, human_move)

            current_player = "ai"
        
        if (board == None):
            break