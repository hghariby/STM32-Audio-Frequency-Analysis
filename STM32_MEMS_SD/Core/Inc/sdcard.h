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

uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc);
uint8_t SD_Test_CMD0(void);
uint8_t SD_Test_CMD8(uint8_t *r7);
uint8_t SD_Init_ACMD41(void);
uint8_t SD_ReadOCR(uint8_t *ocr);
uint8_t SD_ReadSector(uint32_t sector, uint8_t *buffer);
uint8_t SD_WriteSector(uint32_t sector, const uint8_t *buffer);

#define SD_OK     0
#define SD_ERROR  1

uint8_t SD_Init(void);

#endif /* INC_SDCARD_H_ */
