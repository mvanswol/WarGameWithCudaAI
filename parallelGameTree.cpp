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
    // copy current score array, occupancy array, initialize output array and determine number of valid moves
	int * deviceScoreArray;
    int * deviceOccupancyArray;
    int * deviceOutputArray;
    int * outputValues;
    vector<Move> moveList = startingNode->getBoard()->generateMoves(startingNode->getSide());
    int numMoves = moveList.size();

    outputValues = (int *)malloc(numMoves * sizeof(int));

    //allocate device memory
    cudaMalloc((void **)&deviceScoreArray, SQUARED_BOARD * sizeof(int));
    cudaMalloc((void **)&deviceOccupancyArray, SQUARED_BOARD * sizeof(int));
    cudaMalloc((void **)&deviceScoreArray, numMoves * sizeof(int));


    // copy scoreArray and outputArray to the device
    cudaMemcpy(deviceScoreArray, startingNode->getBoard()->scoreArray, SQUARED_BOARD * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOccupancyArray, startingNode->getBoard()->occupancyArray, SQUARED_BOARD * sizeof(int), cudaMemcpyHostToDevice);

    // initialize the outputArray to all 0s
    cudaMemset(deviceOutputArray, 0, numMoves * sizeof(int));

    //Call our parallel algorithm
    callCudaTreeSearch(deviceScoreArray, deviceOccupancyArray, deviceOutputArray, maximizer, numMoves, depth);
    cudaDeviceSynchronize();

    //read output values back after the device has comleted operation
    cudaMemcpy(outputValues, deviceOutputArray, numMoves * sizeof(int), cudaMemcpyDeviceToHost);

    //look for best possible move amongst the output values
    int idx = 0;
    int best = 9999;
    for(int i = 0; i < numMoves; i++)
    {
        if(outputValues[i] < best)
        {
            best = outputArray[i];
            idx = i;
        }
    }

    //free host and device Memory
    free(outputValues);
    cudaFree(deviceScoreArray);
    cudaFree(deviceOccupancyArray);
    cudaFree(deviceOutputArray);

    Move * bestMove = new Move(moveList[idx].getX(), moveList[idx].getY(), maximizer);
    return bestMove;

}
