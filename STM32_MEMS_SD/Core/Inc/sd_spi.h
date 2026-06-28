/*
 * sd_spi.h
 *
 *  Created on: Jun 27, 2026
 *      Author: hasmi
 */

#ifndef INC_SD_SPI_H_
#define INC_SD_SPI_H_

#include "main.h"
#include <stdint.h>

void SD_SPI_Init(void);
void SD_SPI_Select(void);
void SD_SPI_Unselect(void);
uint8_t SD_SPI_TxRx(uint8_t data);
void SD_SPI_SendDummyClocks(void);


#endif /* INC_SD_SPI_H_ */
