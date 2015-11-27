//
//  move.cpp
//  WarGameWithCudaAI
//

#include "move.hpp"

/*
* Constructor for the move class, set all valid params
*/
Move::Move(int x, int y, Side player)
{
    this->x = x;
    this->y = y;
    this->player = player;
}

/*
* Destructor for the move class, we don't need to allocate memory for this
*/
Move::~Move()
{
    
}

/*
* Getter functions for the following moves to be used in board functions during calculations
*/
int Move::getX()
{
    return x;
}

int Move::getY()
{
    return y;
}

Side Move::getPlayer()
{
    return player;
}

