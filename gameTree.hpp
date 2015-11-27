//
//  gameTree.hpp
//  WarGameWithCudaAI
//
//

#ifndef gameTree_hpp
#define gameTree_hpp

#include "common.h"
#include "move.hpp"
#include "board.hpp"
#include "gameTreeNode.hpp"

using namespace std;

class gameTree
{
    
public:
    gameTree(Board * b, Side player);
    ~gameTree();
    
    Move * findBestMove(int depth);
    void searchTree(gameTreeNode * start, int depth);
    
private:
    gameTreeNode * root;
    Side maximizer;
    
};

#endif /* gameTree_hpp */
