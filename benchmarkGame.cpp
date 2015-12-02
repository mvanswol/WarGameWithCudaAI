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
#include "gpuPlayer.h"



int main()
{
    
    srand(time(NULL));
    timespec ts_beg, ts_end;
    Board * game = new Board();
    PlayerAI * firstPlayer = new PlayerAI(PLAYER_ONE, game);
    PlayerAI * secondPlayer = new PlayerAI(PLAYER_TWO, game);
    Move * m;

    Board * gpuGame = new Board();
    // copy over the board to make sure we have the same score
    for(int i = 0; i < SQUARED_BOARD; i++)
    {
      gpuGame->scoreArray[i] = game->scoreArray[i];
      gpuGame->occupancyArray[i] = game->occupancyArray[i];
    }

    GPUPlayer * firstGPU = new GPUPlayer(PLAYER_ONE, gpuGame);
    GPUPlayer * secondGPU = new GPUPlayer(PLAYER_TWO, gpuGame);
    Move * gpuM;



    cout << "Start of CPU Game..." << endl;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_beg);
    while(!game->isGameDone())
    {
        m = firstPlayer->doMove();
        m = secondPlayer->doMove();
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_end);
    cout << "CPU Game completed." << endl;
    cout << (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9 << " sec" << endl;
    
    //cout << game->getScore(PLAYER_ONE) << endl;
    //cout << game->getScore(PLAYER_TWO) << endl;    
    
    cout << "Start of GPU Game..." << endl;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_beg);
    while(!gpuGame->isGameDone())
    {
        gpuM = firstGPU->doMove();
        gpuM = secondGPU->doMove();
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts_end);

    cout << "GPU Game completed." << endl;
    cout << (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9 << " sec" << endl;
    
    //cout << gpuGame->getScore(PLAYER_ONE) << endl;
    //cout << gpuGame->getScore(PLAYER_TWO) << endl; 

    delete gpuGame;
    delete firstGPU;
    delete secondGPU;
    delete game;
    delete firstPlayer;
    delete secondPlayer;
    return 0;
}
