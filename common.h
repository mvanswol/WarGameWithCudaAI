//
//  common.h
//  WarGameWithCudaAI
//

// Common file used to declare defines and other constants used throughout the project

#ifndef common_h
#define common_h


#define BOARD_SIZE 8
#define SQUARED_BOARD (BOARD_SIZE * BOARD_SIZE)
#define DEPTH 1

#define NORTH -1
#define SOUTH 1
#define EAST 1
#define WEST -1

#define BLOCK_SIZE 32

enum Side {PLAYER_ONE = 1, PLAYER_TWO = 2};


#endif /* common_h */
