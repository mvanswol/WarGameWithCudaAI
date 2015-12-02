//
//  move.hpp
//  WarGameWithCudaAI
//

#ifndef move_hpp
#define move_hpp

#include "common.h"

class Move
{

public:
    Move(int x, int y, Side player);
    ~Move();
    int getX();
    int getY();
    Side getPlayer();
    int x, y;
    
private:
    Side player;
};


#endif /* move_hpp */

