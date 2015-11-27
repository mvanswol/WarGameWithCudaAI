//
//  benchmarkGame.cpp
//  WarGameWithCudaAI
//

// File used to get AI bench marks as well as where our main function is located used to debug functions

#include <iostream>
#include <ctime>
#include <vector>
#include "common.h"
#include "board.hpp"
#include "playerAI.hpp"
using namespace std;

int main()
{
    
    srand(time(NULL));
    Board * game = new Board();
    PlayerAI * firstPlayer = new PlayerAI(PLAYER_ONE, game);
    PlayerAI * secondPlayer = new PlayerAI(PLAYER_TWO, game);
    Move * m;
    int turn = 0;
    clock_t timer = clock();
  /*
    for(int j = 0; j < SQUARED_BOARD; j++)
    {
        if(j % 8 == 0)
        {
            cout << endl;
        }
        
        cout << game->scoreArray[j] << " ";
    }
    
    cout << endl;
   */
    
    
    while(!game->isGameDone())
    {
        m = firstPlayer->doMove();
        m = secondPlayer->doMove();
        /*
        for(int i = 0; i < SQUARED_BOARD; i++)
        {
            if ( i % 8 == 0)
            {
                cout << endl;
            }
            cout << game->occupancyArray[i] << " ";
        }
        
        cout << endl;
        */
        
        turn++;
        
    }
    
    
    timer = clock() - timer;
    cout << game->getScore(PLAYER_ONE) << endl;
    cout << game->getScore(PLAYER_TWO) << endl;
    float ms = float(timer) / CLOCKS_PER_SEC * 1000;
    cout << "elasped time is : " << ms << endl;
    
    
    delete game;
    delete firstPlayer;
    delete secondPlayer;
    return 0;
}
