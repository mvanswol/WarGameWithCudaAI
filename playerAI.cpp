//
//  playerAI.cpp
//  WarGameWithCudaAI
//

#include "playerAI.hpp"
#include <ctime>

/*
 * Constructor for the Player class, assign player and opponent values as well as create an empty board
 */
PlayerAI::PlayerAI(Side player, Board * b)
{
    this->player = player;
    this->opponent = (player == PLAYER_ONE) ? PLAYER_TWO : PLAYER_ONE;
    this->board = b;
}
                      
PlayerAI::~PlayerAI()
{
}

Move *PlayerAI::doMove()
{
    clock_t timer = clock();
    gameTree *tree = new gameTree(board, player);
    Move *moveToMake = tree->findBestMove(DEPTH);
    board->doMove(moveToMake);
    timer = clock() - timer;
    float ms = float(timer) / CLOCKS_PER_SEC * 1000;
    cout << "this decision took : " << ms << endl;
    return moveToMake;
}

Board * PlayerAI::getBoard()
{
    return board;
}