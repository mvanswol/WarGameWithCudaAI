//
//  playerAI.hpp
//  WarGameWithCudaAI
//

#ifndef playerAI_hpp
#define playerAI_hpp

#include <iostream>
#include "common.h"
#include "board.hpp"
#include "move.hpp"
#include "gameTree.hpp"
using namespace std;

class PlayerAI {
    
    
public:
    PlayerAI(Side player, Board * b);
    ~PlayerAI();
    Move *doMove();
    Board * getBoard();
    
private:
    Side player;
    Side opponent;
    Board *board;
};

#endif /* playerAI_hpp */
