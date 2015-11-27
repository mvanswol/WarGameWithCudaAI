//
//  gameTree.cpp
//  WarGameWithCudaAI
//


#include "gameTree.hpp"

// The creation and searching of the Game Tree using alpha/Beta prunning and the miniax algorithm

/*
 * Constructor for the game Tree, use the side to determine the maximizer then create a new root node
 */
gameTree::gameTree(Board * b, Side player)
{
    this->maximizer = player;
    root = new gameTreeNode(b, NULL, player);
}

/*
 * Destructor for the Game Tree Node, free the root node we allocated
 */
gameTree::~gameTree()
{
    
}

/*
 * Search the Game Tree to find the best avaliable move at a given depth
 */
Move * gameTree::findBestMove(int depth)
{
    Board *board = root->getBoard();
    vector<Move> moves = board->generateMoves(maximizer);
    gameTreeNode *best = NULL;
    for (int i = 0; i < moves.size(); i++)
    {
        Move *move = new Move(moves[i].getX(), moves[i].getY(), maximizer);
        Board *newBoard = board->copy();
        newBoard->doMove(move);
        gameTreeNode *child = new gameTreeNode(newBoard, move, maximizer);
            
        // pass down alpha and beta values
        child->setAlpha(root->getAlpha());
        child->setBeta(root->getBeta());
            
        // preform search on the child Node
        searchTree(child, depth - 1);
            
        if (best == NULL || child->getBeta() > best->getBeta())
        {
            best = child;
        }
    }
        return best->getMove();
}

void gameTree::searchTree(gameTreeNode *start, int depth)
{
    if (depth == 0)
    {
        start->setAlpha(start->getBoard()->getScore(maximizer));
        start->setBeta(start->getBoard()->getScore(maximizer));
        return;
    }
    
    Board *board = start->getBoard();
    Side otherPlayer = start->getSide() == PLAYER_ONE ? PLAYER_TWO : PLAYER_ONE;
    vector<Move> moves = board->generateMoves(otherPlayer);
    for (int i = 0; i < moves.size(); i++)
    {
        // create the next child
        Move *move = new Move(moves[i].getX(), moves[i].getY(), otherPlayer);
        Board *newBoard = board->copy();
        newBoard->doMove(move);
        gameTreeNode *child = new gameTreeNode(newBoard, move, otherPlayer);
        
        // pass alpha and beta values down
        child->setAlpha(start->getAlpha());
        child->setBeta(start->getBeta());
        
        // search child
        searchTree(child, depth - 1);
        
        if (start->getSide() == maximizer)
        {
            start->setBeta(min(start->getBeta(), child->getAlpha()));
        }
        else
        {
            start->setAlpha(max(start->getAlpha(), child->getBeta()));
        }
        
        delete child;
        
        if (start->getAlpha() > start->getBeta())
        {
            return;
        }
    }
}

