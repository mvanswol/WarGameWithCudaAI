//
//  board.cpp
//  WarGameWithCudaAI
//

// Implementations of the Board Class found Board.h fully commented functions with be provided
#include "board.hpp"


/*
* Make a WAR Game Board of BOARD_SIZE * BOARD_SIZE and initialize square values as well as occupancy state
*/
Board::Board()
{
    for(int i = 0; i < SQUARED_BOARD; i++)
    {
        scoreArray[i] = rand() % 100 + 1;
        occupancyArray[i] = 0;
    }
    
}

/*
 * Desctructor for the board class, we don't allocate any memory so we don't have to free anything here
*/
Board::~Board()
{
    
}

/*
* Copy Constructor for the Board Class
*/
Board * Board::copy()
{
    Board * newBoard = new Board();
    for (int i = 0; i < SQUARED_BOARD; i++)
    {
        newBoard->scoreArray[i] = scoreArray[i];
        newBoard->occupancyArray[i] = occupancyArray[i];
    }
    
    return newBoard;
}

/*
* Function to check and see if there are any open spots left on the board, if so then the game is not done
*/
bool Board::isGameDone()
{
    bool gameDone = true;
    
    for (int i = 0; i < SQUARED_BOARD; i++)
    {
        if(occupancyArray[i] == 0)
        {
            gameDone = false;
        }
    }
    
    return gameDone;
}

/*
* Calculate the score for a given player then return the value
*/
int Board::getScore(Side player)
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
*   Determine whether spot contains an ally, used to calculate Blitzes
*/
bool Board::isAlly(Side player, int x, int y)
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
* Determine whether spot contains an ally, used in blitz calculation
*/
bool Board::isEnemy(Side player, int x, int y)
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
* Determine whether or not a spot is empty
*/
bool Board::isEmpty(int x, int y)
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
* Make a move on the board, update to accurately reflect the board state
*/
void Board::doMove(Move *m)
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
 * Check to see if move is valid
 */
bool Board::isMoveValid(Move *m)
{
    bool valid = true;
    if( m == NULL)
    {
        /* start of game move, null is valid */
        return true;
    }
    else if( occupancyArray[m->getX() + m->getY() * BOARD_SIZE] != 0)
    {
        valid = false;
    }
    
    return valid;
}


/*
* Generate all posssible moves for the current player, note this is just all open spaces
*/
vector<Move> Board::generateMoves(Side player)
{
    vector<Move> possibleMoves;
    for(int i = 0; i < BOARD_SIZE; i++)
    {
        for(int j = 0; j < BOARD_SIZE; j++)
        {
            Move move(i , j, player);
            if (occupancyArray[i + j * BOARD_SIZE] == 0)
            {
                possibleMoves.push_back(move);
            }
        }
    }
    
    return possibleMoves;
}





