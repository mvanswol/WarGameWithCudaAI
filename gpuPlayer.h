#ifndef __GPUPLAYER__H__
#define __GPUPLAYER__H__

#include <iostream>
#include "common.h"
#include "board.hpp"
#include "move.hpp"
#include "parallelGameTree.h"
using namespace std;

class GPUPlayer
{

public:
	GPUPlayer(Side player, Board * b);
	~GPUPlayer();
	Move * doMove();

private:
	Side player;
	Side opponent;
	Board * board;

};

#endif /* gpuPlayer.h */
