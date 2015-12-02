// Implementation of the actual Cuda Search 

#include "cudaGameTree.cuh"


/*
* Constructor for the GPUMove class, set all valid params
*/
CUDA_CALLABLE_MEMBER
GPUMove::GPUMove(int x, int y, Side player)
{
    this->x = x;
    this->y = y;
    this->player = player;
}

/*
* Destructor for the GPUMove class, we don't need to allocate memory for this
*/
CUDA_CALLABLE_MEMBER
GPUMove::~GPUMove()
{
    
}

/*
* Getter functions for the following GPUMoves to be used in board functions during calculations
*/
CUDA_CALLABLE_MEMBER
int GPUMove::getX()
{
    return x;
}

CUDA_CALLABLE_MEMBER
int GPUMove::getY()
{
    return y;
}


CUDA_CALLABLE_MEMBER
Side GPUMove::getPlayer()
{
    return player;
}

/*
*   set the scoreArray and occupancyArray variables to be used in device code
*/
CUDA_CALLABLE_MEMBER GPUBoard::GPUBoard(int * scoreArray, int * occupancyArray)
{
    this->scoreArray = scoreArray;
    this->occupancyArray = occupancyArray;
}

/*
* free the allocated memory
*/
CUDA_CALLABLE_MEMBER GPUBoard::~GPUBoard()
{
}

/*
* Check to see if there is an open spot, if so there is still a valid move left
*/
CUDA_CALLABLE_MEMBER bool GPUBoard::isGameDone()
{
    bool gameDone = true;
    for(int i = 0; i < SQUARED_BOARD; i++)
    {
        if(occupancyArray[i] == 0)
        {
            gameDone = false;
        }
    }

    return gameDone;
}


/*
* Check to see if the move corresponds to an open position
*/
CUDA_CALLABLE_MEMBER bool GPUBoard::isMoveValid(GPUMove *m)
{
    bool validMove = false;

    if(isEmpty(m->getX(), m->getY()))
    {
        validMove = true;
    }

    return validMove;
}

/*
* helper function used in blitz calculations
*/
CUDA_CALLABLE_MEMBER bool GPUBoard::isAlly(Side player, int x, int y)
{
    bool ally = false;
    /* check bounds for the ally */
    if( x < BOARD_SIZE && y < BOARD_SIZE)
    {
        if( player == occupancyArray[x + BOARD_SIZE * y])
        {
            ally = true;
        }
    }
    
    return ally;
}

/*
* helper function used in blitz calculations
*/
CUDA_CALLABLE_MEMBER bool GPUBoard::isEnemy(Side player, int x, int y)
{
    bool enemy = false;
    /* check spot boundary */
    if ( x < BOARD_SIZE && y < BOARD_SIZE)
    {
        if( player == PLAYER_ONE && occupancyArray[x + BOARD_SIZE * y] == PLAYER_TWO)
        {
            enemy = true;
        }
        else if(player == PLAYER_TWO && occupancyArray[x + BOARD_SIZE * y] == PLAYER_ONE)
        {
            enemy = true;
        }
    }
    
    return enemy;
}

/*
* Check to see if a spot is open
*/
CUDA_CALLABLE_MEMBER bool GPUBoard::isEmpty(int x, int y)
{
    bool empty = false;
    if( x < BOARD_SIZE && y < BOARD_SIZE)
    {
        if( occupancyArray[x + y * BOARD_SIZE] == 0)
        {
            empty = true;
        }
    }
    
    return empty;
}

/*
* We've decided on a particular move, now do that move
*/
CUDA_CALLABLE_MEMBER void GPUBoard::doMove(GPUMove * m)
{
    /* 
     * There are two types of moves we must take into account
     * 1. The Paradrop, this just occupies a given spot, easy move
     * 2. The Blitz, this takes an adjacent spot and also converts enemies in adjacent squares
     *      We need to check for this move, and convert enemies if that is the case
     */
    
    // first step determine if we have an adjacent ally, this will determine if we have a blitz
    bool isBlitz = false;
    Side playerSide = m->getPlayer();
    int x = m->getX();
    int y = m->getY();
    isBlitz = (isAlly(playerSide, x + NORTH, y) || isAlly(playerSide, x + SOUTH, y) || isAlly(playerSide, x, y + EAST) || isAlly(playerSide, x, y + WEST));
    
    if(isBlitz)
    {
        //We have a blitz move check if there are adjacent enemies and capture them
        occupancyArray[x + y * BOARD_SIZE] = playerSide;
        bool enemy = false;
        enemy = isEnemy(playerSide, x + NORTH, y);
        if ( enemy)
        {
            occupancyArray[(x+NORTH) + y * BOARD_SIZE] = playerSide;
        }
        enemy = isEnemy(playerSide, x + SOUTH, y);
        if ( enemy)
        {
            occupancyArray[(x+SOUTH) + y * BOARD_SIZE] = playerSide;
        }
        enemy = isEnemy(playerSide, x, y + EAST);
        if ( enemy)
        {
            occupancyArray[x + (y+EAST) * BOARD_SIZE] = playerSide;
        }
        enemy = isEnemy(playerSide, x, y + WEST);
        if ( enemy)
        {
            occupancyArray[x + (y+WEST) * BOARD_SIZE] = playerSide;
        }
    }
    else
    {
        // Just a paradrop claim spot and that's it
        occupancyArray[x + y * BOARD_SIZE] = playerSide;
    }
}

/*
* calculate the zero sum score, add up the spots we have, subtract the spots the opponents own
*/
CUDA_CALLABLE_MEMBER int GPUBoard::getScore(Side player)
{
    int playerScore = 0;
    for (int i = 0; i < SQUARED_BOARD; i++)
    {
        if(player == occupancyArray[i])
        {
            playerScore += scoreArray[i];
        }
        else if(player != occupancyArray[i] && occupancyArray[i] != 0)
        {
            /* we need to subtract the opponents score */
            playerScore -= scoreArray[i];
        }
    }
    
    return playerScore;
}


/*
* Constructor for the GPU Node set our default values
*/
CUDA_CALLABLE_MEMBER GPUNode::GPUNode(GPUBoard * b, GPUMove * m, Side maximizer)
{
    this->board = b;
    this->m = m;
    this->maximizer = maximizer;
    this->alpha = 9999;
    this->beta = -9999;
}

/*
* Destructor for the GPUNode, delete the board state associated with this corresponding node
*/
CUDA_CALLABLE_MEMBER GPUNode::~GPUNode()
{
    //free board state associated with this particular node
    delete board;
}

/*
* Return the board state associated with this particular node
*/
CUDA_CALLABLE_MEMBER GPUBoard * GPUNode::getBoard()
{
    return board;
}

/*
* return the particular move we are exploring at this node
*/
CUDA_CALLABLE_MEMBER GPUMove * GPUNode::getMove()
{
    return m;
}

/*
* Return the side this node is maximizing
*/
CUDA_CALLABLE_MEMBER Side GPUNode::getSide()
{
    return maximizer;
}


/*
* Return the parent Node associated with this node
*/
CUDA_CALLABLE_MEMBER GPUNode * GPUNode::getParent()
{
    return parent;
}

/*
* Set the Parent Node of the current Node we are operating on
*/
CUDA_CALLABLE_MEMBER void GPUNode::setParent(GPUNode * node)
{
    this->parent = node;
}

/*
* Get the current alpha value of our node
*/
CUDA_CALLABLE_MEMBER int GPUNode::getAlpha()
{
    return alpha;
}

/*
* Get the current beta value
*/
CUDA_CALLABLE_MEMBER int GPUNode::getBeta()
{
    return beta;
}

/*
* Set the alpha value of this node
*/
CUDA_CALLABLE_MEMBER void GPUNode::setAlpha(int alpha)
{
    this->alpha = alpha;
}

/*
* Set the beta value of this node
*/
CUDA_CALLABLE_MEMBER void GPUNode::setBeta(int beta)
{
    this->beta = beta;
}


/*
* Helper function for searching the tree, purely on the device to take advantage of recursion
*/
__device__ void cudaTreeSearchHelper(GPUNode * node, Side player, Side maximizer, int depth)
{
	/* Get board state so that we can find optimal moves */
	GPUBoard * currBoard = node->getBoard();
	Side opponent = (player == PLAYER_ONE) ? PLAYER_TWO : PLAYER_ONE;

	// if depth equal 0, calculate scores and then return
	if (depth == 0) 
	{
       node->setAlpha(node->getBoard()->getScore(maximizer));
       node->setBeta(node->getBoard()->getScore(maximizer));
       return;
    }

    // Else try to computation of all other possible moves in this game state
    for(int i = 0; i < BOARD_SIZE; i++)
    {
    	for(int j = 0; j < BOARD_SIZE; j++)
    	{
    		GPUMove * move = new GPUMove(i,j, opponent);
    		if(currBoard->isMoveValid(move))
    		{
    			// create a new board and preform the move
    			GPUBoard * newBoard = new GPUBoard(currBoard->scoreArray,currBoard->occupancyArray);
    			newBoard->doMove(move);
    			GPUNode * child = new GPUNode(newBoard, move, opponent);

    			//pass down alpha and beta values
    			child->setAlpha(node->getAlpha());
    			child->setBeta(node->getBeta());

    			// continue to search down this path
    			cudaTreeSearchHelper(child, opponent, maximizer, depth - 1);

    			// check out heuristic
    			if(player == maximizer)
    			{
    				node->setBeta(min(node->getBeta(), child->getAlpha()));
    			}
    			else
    			{
    				node->setAlpha(max(node->getAlpha(), child->getBeta()));
    			}

    			delete child;
    			delete newBoard;

    			// check if we have to prune this node
    			if(node->getAlpha() >= node->getBeta())
    			{
    				return;
    			}

    		}

    	}
    }
}


/*
* Initial call to the tree search, which calls our device helper function to do the rest of the work
*/
__global__ void cudaTreeSearch(Move * moveList, int * scoreArray, int * occupancyArray, int * outputValues, Side player, Side maximizer, int alpha, int beta, int numMovesLeft, int depth)
{
	// only allow the first thread of each block to do high-level tasks
	if(threadIdx.x == 0)
	{
		// make only one node per block
		GPUMove * move  = new GPUMove(moveList[blockIdx.x].x, moveList[blockIdx.x].y, player);

		/* make a new board and do move on that board */
		GPUBoard * newBoard = new GPUBoard(scoreArray, occupancyArray);
		newBoard->doMove(move);
		GPUNode * node = new GPUNode(newBoard, move, player);

		/* set the alpha and beta values, we've learned from cpu side */
		node->setAlpha(alpha);
		node->setBeta(beta);

		/* preform the rest of the search */
		cudaTreeSearchHelper(node, player, maximizer, depth);

		//update the values array, if parent node is maximizer look at child alpha
		if(player == maximizer)
		{
			outputValues[blockIdx.x] = node->getBeta();
		}
		else
		{
			outputValues[blockIdx.x] = node->getAlpha();
		}

		delete newBoard;
		delete move;
        delete node;
	}
}

/*
* Helper function to call the kernel code
*/
void callCudaTreeSearch(Move * moveList, int * scoreArray, int * occupancyArray, int * outputValues, Side player, Side maximizer, int alpha, int beta, int numMovesLeft, int depth)
{
	cudaTreeSearch<<<numMovesLeft, BLOCK_SIZE>>>(moveList, scoreArray, occupancyArray, outputValues, player, maximizer, alpha, beta, numMovesLeft, depth);
}
