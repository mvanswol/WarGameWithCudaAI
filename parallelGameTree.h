#ifndef __PARALLELGAMETREE__H__
#define __PARALLELGAMETREE__H__

// File Includes
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "common.h"
#include "board.hpp"
#include "move.hpp"
#include "gameTreeNode.hpp"
#include "cudaGameTree.cuh"

// Ideas for the parallel computation inspired by the conference
// http://on-demand.gputechconf.com/gtc/2010/presentations/S12207-Playing-Zero-Sum-Games-on-the-GPU.pdf

class ParallelGameTree
{

public:
	ParallelGameTree(Board * b, Side player);
	~ParallelGameTree();
	gameTreeNode * getRoot();
	Move * searchTree(gameTreeNode * startingNode, int depth);

private:
	gameTreeNode * root;
	Side maximizer;
};



#endif /* parallelGameTree.h */

