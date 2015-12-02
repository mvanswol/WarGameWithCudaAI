// Implementation of the parallel game tree
#include "parallelGameTree.h"


//function to make sure that our gpu code works without error
/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
*/
/*
* Default constructor for the Game Tree, set maximizing player and create a root node
*/
ParallelGameTree::ParallelGameTree(Board * b, Side player)
{
	this->maximizer = player;
	this->root = new gameTreeNode(b, NULL, player);
}

/*
* ParallelGameTree destructor
*/
ParallelGameTree::~ParallelGameTree()
{
	// freeing the root is taken care of elsewhere in the code
	delete root;
}


/*
* Get the value of the root node
*/
gameTreeNode * ParallelGameTree::getRoot()
{
	return root;
}


/*
* Search the Parallel Game tree using the Principal Variation Splitting Technique
*/
Move * ParallelGameTree::searchTree(gameTreeNode * startingNode, int depth)
{
	// We've reached the end set the appropriate alpha and beta values
	if (depth == 0) 
	{
		startingNode->setAlpha(startingNode->getBoard()->getScore(maximizer));
		startingNode->setBeta(startingNode->getBoard()->getScore(maximizer));
		return NULL;
	}

	// get all possible moves on this board state for the opponent
	Board * currBoard = startingNode->getBoard();
	Side opponent = startingNode->getSide() == PLAYER_ONE ? PLAYER_TWO : PLAYER_ONE;
	vector<Move> moves = currBoard->generateMoves(opponent);

	// make sure that at least one move exists
	if (moves.size() == 0) 
	{
    	return NULL;
    }

    // In doing the PV split technique we allow the CPU to do searches on the first child node
    Move * move = new Move(moves[0].getX(), moves[0].getY(), opponent);
    Board * newBoard = currBoard->copy();
    newBoard->doMove(move);
    gameTreeNode * child = new gameTreeNode(newBoard, move, opponent);

    // pass alpha and beta values down
    child->setAlpha(startingNode->getAlpha());
    child->setBeta(startingNode->getBeta());

    Move * best = searchTree(child, depth - 1);

    //Prepare the GPU computation by first loading values of interest
    // we will do this by first getting the gpu computed alpha/beta values at the particular level
    int * values;
    values = (int *)calloc(moves.size(), sizeof(int));

    if(startingNode->getSide() == maximizer)
    {
    	startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
        values[0] = child->getAlpha();
    }
    else
    {
    	//Compute the alpha values between current node and the child
    	startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
        values[0] = child->getBeta();
    }

    // We are done with the child node, so we delete it to avoid memory leaks
    delete child;

    /* 
    * We've finished with the CPU side of computation now we must prepare 
    * for the rest of the children on the cpu, first load initial values
    */
    int movesLeft = moves.size() - 1;
    Move * deviceMove;
    Move * movePtr = &moves[1]; // used to copy the array over
    int * deviceScoreArray; // used to copy over current game state
    int * deviceOccupancyArray;
    int * deviceValues; // used to store computations in the array, for alpha/beta values


    //Begin to copy over the values onto the device, make sure to check for errors
    cudaMalloc((void **) &deviceMove, movesLeft * sizeof(Move));
    cudaMalloc((void **) &deviceScoreArray, BOARD_SIZE * BOARD_SIZE * sizeof(int));
    cudaMalloc((void **) &deviceOccupancyArray, BOARD_SIZE * BOARD_SIZE * sizeof(int));
    cudaMalloc((void **) &deviceValues, movesLeft * sizeof(int));

    cudaMemcpy(deviceScoreArray, currBoard->scoreArray, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOccupancyArray, currBoard->occupancyArray, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMove, movePtr, movesLeft * sizeof(Move), cudaMemcpyHostToDevice);

    cudaMemset(deviceValues, 0, movesLeft * sizeof(int)); // set all of the values into the to 0

    // call the kernel function to search the rest of the tree in parallel
    callCudaTreeSearch(deviceMove, deviceScoreArray, deviceOccupancyArray, deviceValues, opponent,
    	maximizer, startingNode->getAlpha(), startingNode->getBeta(), movesLeft, depth - 1);

    // copy remaining child values into host array
    cudaMemcpy(values + 1, deviceValues, movesLeft * sizeof(int), cudaMemcpyDeviceToHost);

    // find the best move
    int idx = 0;
    if (startingNode->getSide() == maximizer) 
    {
    	int best = 9999;
    	for (int i = 0; i <= movesLeft; i++) 
    	{
    		if (values[i] < best) 
    		{
    			best = values[i];
    			idx = i;
    		}
    	}
    	startingNode->setBeta(best);
    } 
    else 
    {
    	int best = -9999;
    	for (int i = 0; i <= movesLeft; i++) 
    	{
    		if (values[i] > best) 
    		{
    			best = values[i];
    			idx = i;
    		}
    	}
    	startingNode->setAlpha(best);
    }

    // Free device allocated memory
    cudaFree(deviceValues);
    cudaFree(deviceScoreArray);
    cudaFree(deviceOccupancyArray);
    cudaFree(deviceMove);

    // return the current best move
    Move *curMove = new Move(moves[idx].getX(), moves[idx].getY(), maximizer);
    return curMove;

}
