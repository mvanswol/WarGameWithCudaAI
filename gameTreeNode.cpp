//
//  gameTreeNode.cpp
//  WarGameWithCudaAI
//

#include "gameTreeNode.hpp"

/*
 * Constructor for our gameTreeNode, set values and then give a default alpha/beta
 */
gameTreeNode::gameTreeNode(Board * b, Move * m, Side maximizer)
{
    this->board = b;
    this->move = m;
    this->maximizer = maximizer;
    this->alpha = 9999;
    this->beta = -9999;
}

/*
 * Destructor for the gameTreeNode, free the dynamically allocated board
 */
gameTreeNode::~gameTreeNode()
{
    delete board;
}

/*
 * What follows here are some pretty standard getter and setter functions
 */

Board * gameTreeNode::getBoard()
{
    return board;
}

Move * gameTreeNode::getMove()
{
    return move;
}

gameTreeNode * gameTreeNode::getParent()
{
    return parent;
}

Side gameTreeNode::getSide()
{
    return maximizer;
}

void gameTreeNode::setParent(gameTreeNode * node)
{
    parent = node;
}

int gameTreeNode::getAlpha()
{
    return alpha;
}

int gameTreeNode::getBeta()
{
    return beta;
}

void gameTreeNode::setAlpha(int alpha)
{
    this->alpha = alpha;
}

void gameTreeNode::setBeta(int beta)
{
    this->beta = beta;
}
