//
//  gameTreeNode.hpp
//  WarGameWithCudaAI
//

#ifndef gameTreeNode_hpp
#define gameTreeNode_hpp


#include <vector>
#include "common.h"
#include "board.hpp"
#include "move.hpp"


class gameTreeNode
{
    
public:
    gameTreeNode(Board * b, Move * m, Side maximizer);
    ~gameTreeNode();
    Board * getBoard();
    Move * getMove();
    Side getSide();
    gameTreeNode * getParent();
    void setParent(gameTreeNode * node);
    int getAlpha();
    int getBeta();
    void setAlpha(int alpha);
    void setBeta(int beta);
    
private:
    Board * board;
    Move * move;
    Side maximizer;
    int alpha;
    int beta;
    gameTreeNode * parent;
    
    
};

#endif /* gameTreeNode_hpp */
