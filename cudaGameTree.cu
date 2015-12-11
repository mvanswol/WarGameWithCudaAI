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
    
    printf("score : %d\n", playerScore);
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
    this->alpha = 999999;
    this->beta = -999999;
}

/*
* Destructor for the GPUNode, delete the board state associated with this corresponding node
*/
CUDA_CALLABLE_MEMBER GPUNode::~GPUNode()
{
    //free board state associated with this particular node
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


__device__ void cudaTreeSearchHelper(GPUNode * node, Side lastPlayer, Side maximizer, int depth)
{
    //check to see if we have reached the bottom
    if(depth == 0)
    {
        //return the heuristic of the node
        node->setAlpha(node->getBoard()->getScore(maximizer));
       node->setBeta(node->getBoard()->getScore(maximizer));
       return;
    }

    //otherwise try all possible combinations of moves
    for(int i = 0; i < BOARD_SIZE; i++)
    {
        for(int j = 0; j < BOARD_SIZE; j++)
        {
            Side currPlayer = (lastPlayer == PLAYER_ONE) ? PLAYER_TWO : PLAYER_ONE;
            GPUMove * currMove = new GPUMove(i,j,currPlayer);
            if(node->getBoard()->isMoveValid(currMove))
            {
                GPUBoard * gameState = new GPUBoard(node->getBoard()->scoreArray, node->getBoard()->occupancyArray);
                gameState->doMove(currMove);
                GPUNode * child = new GPUNode(gameState, currMove, currPlayer);

                //pass down alpha and beta
                child->setAlpha(node->getAlpha());
                child->setBeta(node->getBeta());

                //continue the search
                cudaTreeSearchHelper(child, currPlayer, maximizer, depth - 1);

                //evaulate based upon heuristic
//                if(node->getSide() == maximizer)
//                {
                    node->setBeta(max(node->getBeta(), child->getAlpha()));
//                }
//                else
//                {
                    node->setAlpha(min(node->getAlpha(), child->getBeta()));
//                }
            }
        }
    }
}


__device__ void cudaTreeSearchThread(GPUNode * node, Side lastPlayer, Side maximizer,  int depth)
{
    //check to see if we've reached a depth of 0
    if(depth == 0)
    {
        //return the heuristic of the node
        node->setAlpha(node->getBoard()->getScore(maximizer));
       node->setBeta(node->getBoard()->getScore(maximizer));
       return;
    }

    //extract move from thread index
    int x = threadIdx.x % BOARD_SIZE;
    int y = threadIdx.x / BOARD_SIZE;
    Side currPlayer = (lastPlayer == PLAYER_ONE) ? PLAYER_TWO : PLAYER_ONE;
    GPUMove * currMove = new GPUMove(x,y, currPlayer);

    //check to see if the move is legal, if so we have to move down deeper
    if(node->getBoard()->isMoveValid(currMove))
    {
        //create a new board and child node for the move
        GPUBoard * gameState = new GPUBoard(node->getBoard()->scoreArray, node->getBoard()->occupancyArray);
        gameState->doMove(currMove);
        GPUNode * child = new GPUNode(gameState, currMove, currPlayer);

        //pass down alpha and beta
        child->setAlpha(node->getAlpha());
        child->setBeta(node->getBeta());

        //continue the search
        cudaTreeSearchHelper(child, currPlayer, maximizer, depth - 1);

        //evaulate based upon heuristic
//        if(node->getSide() == maximizer)
//        {
            node->setBeta(max(node->getBeta(), child->getAlpha()));
//        }
//        else
//        {
            node->setAlpha(min(node->getAlpha(), child->getBeta()));
//        }
    }
}


__global__ void cudaTreeSearchBlock(int * scoreArray, int * occupancyArray, int * outputArray, Side maximizer, int numMoves, int depth)
{
    // have only the first thread of each block start a search, then at the next level all of the other threads will join in
        // calculate board position with the given blockIdx
        int x = blockIdx.x % BOARD_SIZE;
        int y = blockIdx.x / BOARD_SIZE;

        //make a move on the given board state
        GPUBoard * gameState = new GPUBoard(scoreArray, occupancyArray);
        GPUMove * currMove = new GPUMove(x,y, maximizer);
        gameState->doMove(currMove);
        GPUNode * node = new GPUNode(gameState, currMove, maximizer);

        cudaTreeSearchHelper(node, maximizer, maximizer, depth - 1);

        //set out array to the beta value since this will always be evaulated as the maximizer
        if(threadIdx.x == 0)
        {
             outputArray[blockIdx.x] = node->getBeta();
        }

        //clean up
        delete gameState;
        delete currMove;
        delete node;
}


/*
* Helper function to call the kernel code
*/
void callCudaTreeSearch(int * scoreArray, int * occupancyArray, int * outputArray, Side maximizer, int numMoves, int depth)
{
	cudaTreeSearchBlock<<<numMoves, 1>>>(scoreArray, occupancyArray, outputArray, maximizer, numMoves, depth);
}
