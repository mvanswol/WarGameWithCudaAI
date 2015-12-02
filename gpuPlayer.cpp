// Implementation of the GPUPlayer

#include "gpuPlayer.h"

/*
* Constructor for the GPU Player AI, set player, opponent and the board value
*/
GPUPlayer::GPUPlayer(Side player, Board * b)
{
	this->player = player;
	this->opponent = (player == PLAYER_ONE) ? PLAYER_TWO : PLAYER_ONE;
	this->board = b;
}

/*
* Destructor for the GPUPlayer, we don't allocate any memory for the struct we don't need any cleanup
*/ 
GPUPlayer::~GPUPlayer()
{

}

/*
* Do Move function, we create a parallel decision tree and see what is the best move according to our algorithm
*/
Move * GPUPlayer::doMove()
{
	ParallelGameTree *tree = new ParallelGameTree(board, player);
    Move *moveToMake = tree->searchTree(tree->getRoot(), DEPTH);
    board->doMove(moveToMake);
    return moveToMake;
}

