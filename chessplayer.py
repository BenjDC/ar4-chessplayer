import cv2
import sys
import numpy as np
import argparse
from vision import vision
from chessai import chessmonitor
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="AR4 Chess Player – vision + robot"
    )

    # Paramètre couleur du bras
    parser.add_argument(
        "--color",
        choices=["blancs", "noirs"],
        default="noirs",
        help="Couleur jouée par le bras (blancs | noirs). Noirs par défaut "
    )

    # Paramètre mode debug vision
    parser.add_argument(
        "--debug-vision",
        action="store_true",
        help="Active les affichages et images intermédiaires."
    )

    # Mode test : vérifie la détection à partir d'images
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="fournit une image d'essai pour tester la détection (pas de camera necessaire)."
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("=== AR4 Chess Player ===")
    
    fishy = chessmonitor.Chessfish()

    device = "cpu"
    model = vision.load_chess_model("vision/model_chess.pth", device)
    
    if (not (args.test == None)):
        boardstate = vision.get_board(args.test)
        vision.predict_board_occupancy(boardstate, model, device, debug=True)
    else:
        if args.color == "blancs":
            current_player = "ai"
        else:
            current_player = "human"
        
        print(f"Je joue avec les {args.color}")

        while(True):
            
            if current_player == "ai":
                print("Tour IA")

                move = fishy.get_ai_move()

                current_player = "human"

                # lorsque le robot sera complètement opérationnel, il faudra introduire l'appel au bras ici ! 
                # pour le moment c'est à l'humain de joué le coup à sa place
                
                
            else:
                print("Tour Humain")

                current_player = "ai"

            player_input = input("Confirmer lorsque le coup est joué")

            if player_input == "a":
                print("abandon session")
                break


            after_move = vision.get_board()

            after_move_occupancy = vision.predict_board_occupancy(after_move, model, debug=True)

            move = vision.detect_move_from_occupancy(fishy.get_fen(), after_move_occupancy)

            

            fishy.play_move(move)
