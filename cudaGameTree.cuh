#ifndef __CUDA__GAMETREE__CUH__
#define __CUDA__GAMETREE__CUH__

#include <cstdio>
#include "common.h"
#include "move.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class GPUMove
{

public:
    CUDA_CALLABLE_MEMBER GPUMove(int x, int y, Side player);
    CUDA_CALLABLE_MEMBER ~GPUMove();
    CUDA_CALLABLE_MEMBER int getX();
    CUDA_CALLABLE_MEMBER int getY();
    CUDA_CALLABLE_MEMBER Side getPlayer();
    
private:
    int x, y;
    Side player;
};
   


class GPUBoard
{
public:
	CUDA_CALLABLE_MEMBER GPUBoard(int * scoreArray, int * occupancyArray);
	CUDA_CALLABLE_MEMBER ~GPUBoard();
	CUDA_CALLABLE_MEMBER bool isGameDone();
	CUDA_CALLABLE_MEMBER bool isMoveValid(GPUMove *m);
	CUDA_CALLABLE_MEMBER void doMove(GPUMove * m);
	CUDA_CALLABLE_MEMBER int getScore(Side player);
	int * scoreArray;
	int * occupancyArray;

private:
	CUDA_CALLABLE_MEMBER bool isAlly(Side player, int x, int y);
	CUDA_CALLABLE_MEMBER bool isEnemy(Side player, int x, int y);
	CUDA_CALLABLE_MEMBER bool isEmpty(int x, int y);
};



class GPUNode
{

public:
	CUDA_CALLABLE_MEMBER GPUNode(GPUBoard * b, GPUMove * m, Side maximizer);
	CUDA_CALLABLE_MEMBER ~GPUNode();
	CUDA_CALLABLE_MEMBER GPUBoard * getBoard();
	CUDA_CALLABLE_MEMBER GPUMove * getMove();
	CUDA_CALLABLE_MEMBER Side getSide();
	CUDA_CALLABLE_MEMBER GPUNode * getParent();
	CUDA_CALLABLE_MEMBER void setParent(GPUNode * node);
	CUDA_CALLABLE_MEMBER int getAlpha();
	CUDA_CALLABLE_MEMBER int getBeta();
	CUDA_CALLABLE_MEMBER void setAlpha(int alpha);
	CUDA_CALLABLE_MEMBER void setBeta(int beta);

private:
	GPUBoard * board;
	GPUMove * m;
	Side maximizer;
	int alpha;
	int beta;
	GPUNode * parent;

};

void callCudaTreeSearch(int * scoreArray, int * occupancyArray, int * outputArray, Side maximizer, int numMoves, int depth);


#endif /* cudaGameTree.cuh */

