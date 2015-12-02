//
//  board.h
//  WarGameWithCudaAI
//
// Initial implementation of the Board Class used for the War Game AI

// Game Description can be found http://web.engr.illinois.edu/~slazebni/spring15/assignment2.html


#ifndef board_h
#define board_h

// Includes for this file
#include <vector>
#include "common.h"
#include "move.hpp"

using namespace std;


class Board
{

public:
    Board();
    ~Board();
    Board * copy();
    bool isGameDone();
    bool isMoveValid(Move *m);
    vector<Move> generateMoves(Side player);
    void doMove(Move *m);
    int getScore(Side Player);
    
    int scoreArray[BOARD_SIZE * BOARD_SIZE];
    int occupancyArray[BOARD_SIZE * BOARD_SIZE];
    
private:
    bool isAlly(Side player, int x , int y);
    bool isOpen(int x, int y);
    bool isEnemy(Side player, int x, int y);
    bool isEmpty(int x, int y);
};





#endif /* board_h */
