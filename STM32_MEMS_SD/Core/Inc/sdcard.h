/*
 * sdcard.h
 *
 *  Created on: Jun 27, 2026
 *      Author: hasmi
 */

#ifndef INC_SDCARD_H_
#define INC_SDCARD_H_

#include "main.h"
#include <stdint.h>

#define SD_OK     0
#define SD_ERROR  1

uint8_t SD_Init(void);
uint8_t SD_ReadSector(uint32_t sector, uint8_t *buffer);
uint8_t SD_WriteSector(uint32_t sector, const uint8_t *buffer);

#endif /* INC_SDCARD_H_ */
