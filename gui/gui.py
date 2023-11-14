import pygame
import chess
import math
import random

# display size
X = 800
Y = 800

scrn = pygame.display.set_mode((X, Y))
pygame.init()

#basic colours
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)

#initialise chess board
b = chess.Board()

#load piece images
pieces = {
    'p': pygame.transform.scale(pygame.image.load('./images/b_pawn.png'), (100, 100)),
    'n': pygame.transform.scale(pygame.image.load('./images/b_knight.png'), (100, 100)),
    'b': pygame.transform.scale(pygame.image.load('./images/b_bishop.png'), (100, 100)),
    'r': pygame.transform.scale(pygame.image.load('./images/b_rook.png'), (100, 100)),
    'q': pygame.transform.scale(pygame.image.load('./images/b_queen.png'), (100, 100)),
    'k': pygame.transform.scale(pygame.image.load('./images/b_king.png'), (100, 100)),
    'P': pygame.transform.scale(pygame.image.load('./images/w_pawn.png'), (100, 100)),
    'N': pygame.transform.scale(pygame.image.load('./images/w_knight.png'), (100, 100)),
    'B': pygame.transform.scale(pygame.image.load('./images/w_bishop.png'), (100, 100)),
    'R': pygame.transform.scale(pygame.image.load('./images/w_rook.png'), (100, 100)),
    'Q': pygame.transform.scale(pygame.image.load('./images/w_queen.png'), (100, 100)),
    'K': pygame.transform.scale(pygame.image.load('./images/w_king.png'), (100, 100)),
}


def update(scrn,board):
    '''
    updates the screen basis the board class
    '''
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            pass
        else:
            scrn.blit(pieces[str(piece)],((i%8)*100,700-(i//8)*100))
    
    for i in range(7):
        i=i+1
        pygame.draw.line(scrn,WHITE,(0,i*100),(800,i*100))
        pygame.draw.line(scrn,WHITE,(i*100,0),(i*100,800))

    pygame.display.flip()

def main(BOARD):
    '''
    human vs human game
    '''
    scrn.fill((222, 184, 136))
    # name window
    pygame.display.set_caption('Chess')

    # variable to be used later
    index_moves = []

    clock = pygame.time.Clock()  # create a clock object

    status = True
    while status:
        for event in pygame.event.get():
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False

            # if mouse clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                # remove previous highlights
                scrn.fill((222, 184, 136))
                # get position of mouse
                pos = pygame.mouse.get_pos()

                # find which square was clicked and index of it
                square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                index = (7 - square[1]) * 8 + (square[0])

                # if we are moving a piece
                if index in index_moves:
                    move = moves[index_moves.index(index)]
                    BOARD.push(move)

                    # reset index and moves
                    index = None
                    index_moves = []

                # highlight possible moves
                else:
                    # check the square that is clicked
                    piece = BOARD.piece_at(index)
                    # if empty pass
                    if piece is None:
                        pass
                    else:
                        # figure out what moves this piece can make
                        all_moves = list(BOARD.legal_moves)
                        moves = []
                        for m in all_moves:
                            if m.from_square == index:
                                moves.append(m)

                                t = m.to_square

                                TX1 = 100 * (t % 8)
                                TY1 = 100 * (7 - t // 8)

                                # highlight squares it can move to
                                pygame.draw.rect(scrn, GREY, pygame.Rect(TX1, TY1, 100, 100), 5)

                        index_moves = [a.to_square for a in moves]

        # update screen
        update(scrn, BOARD)

        # control the frame rate
        clock.tick(30)

    # deactivates the pygame library
    pygame.quit()


def main_one_agent(BOARD, agent):
    '''
    agent vs human game
    '''
    agent_color = random.choice([True, False])
    print(agent_color)
    # make background chess.com default color
    scrn.fill((222, 184, 136))

    pygame.display.set_caption('Chess')

    # variable to be used later
    index_moves = []

    status = True
    while status:
        # update screen
        update(scrn, BOARD)

        if BOARD.turn == agent_color:
            move = agent(BOARD)
            BOARD.push(move)
            scrn.fill((222, 184, 136))

        else:
            for event in pygame.event.get():
                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT:
                    status = False

                if event.type == pygame.MOUSEBUTTONDOWN:

                    scrn.fill((222, 184, 136))

                    # get pos of mouse
                    pos = pygame.mouse.get_pos()

                    # find which square was clicked and index of it
                    square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                    index = (7 - square[1]) * 8 + (square[0])

                    # if we have already highlighted moves and are making a move
                    if index in index_moves:
                        move = moves[index_moves.index(index)]
                        BOARD.push(move)
                        index = None
                        index_moves = []

                    # highlight possible moves
                    else:
                        piece = BOARD.piece_at(index)

                        if piece is None:
                            pass
                        else:
                            all_moves = list(BOARD.legal_moves)
                            moves = []
                            for m in all_moves:
                                if m.from_square == index:
                                    moves.append(m)

                                    t = m.to_square

                                    TX1 = 100 * (t % 8)
                                    TY1 = 100 * (7 - t // 8)

                                    pygame.draw.rect(scrn, BLUE, pygame.Rect(TX1, TY1, 100, 100), 5)

                            index_moves = [a.to_square for a in moves]

        # deactivates the pygame library
        if BOARD.outcome() is not None:
            print(BOARD.outcome())
            status = False
            print(BOARD)

    pygame.quit()

def main_two_agent(BOARD,agent1,agent2):
    '''
    for agent vs agent game
    
    '''
    agent1_color = random.choice([True, False])
    # make background light brown (chess.com default color)
    scrn.fill((222, 184, 136))

    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later

    status = True
    while (status):
        #update screen
        update(scrn,BOARD)
        
        if BOARD.turn==agent1_color:
            BOARD.push(agent1(BOARD))

        else:
            BOARD.push(agent2(BOARD))

        scrn.fill(BLACK)
            
        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()

main(b)
#main_one_agent(b, agent)