import cv2
import numpy as np
from vision import vision, detection
from chessai import chessmonitor

if __name__ == "__main__":


    board, fishplayer = chessmonitor.init_chess()
    

    while(True):
        
        initial = vision.get_board()
        
        input("Confirmer lorsque ton coup est joué")

        final = vision.get_board()

        state_before, masks_before = vision.board_state(initial)
        state_after,  masks_after  = vision.board_state(final)

        white_move = vision.detect_move_from_state(state_before, state_after)

        vision.debug_silhouettes(masks_before, masks_after)

        board = chessmonitor.play_move(white_move, board, fishplayer)

        if (board == None):
            break
        else:
            input("Confirmer lorsque mon coup est joué")


        initial = final