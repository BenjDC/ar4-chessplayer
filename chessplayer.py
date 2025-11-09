import cv2
import numpy as np
from vision import vision
from chessai import chessmonitor

if __name__ == "__main__":


    board, fishplayer = chessmonitor.init_chess()
    

    while(True):
        
        initial = vision.get_board()

        vision.debug_grid(initial)

        input("Confirmer lorsque ton coup est joué")

        final = vision.get_board()

        state_before, masks_before = vision.board_state(initial)
             
        #vision.show_occupancy_grid(state_before)

        state_after,  masks_after  = vision.board_state(final)

        white_move = vision.detect_move_from_state(state_before, state_after)

        board = chessmonitor.play_move(white_move, board, fishplayer)

        if (board == None):
            break
        else:
            input("Confirmer lorsque mon coup est joué")


        initial = final