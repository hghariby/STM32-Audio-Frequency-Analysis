/*
 * sd_spi.c
 *
 *  Created on: Jun 27, 2026
 *      Author: hasmi
 */


#include "sd_spi.h"
extern SPI_HandleTypeDef hspi2;

void SD_SPI_Init(void)
{
    SD_SPI_Unselect();
    SD_SPI_SendDummyClocks();
}

void SD_SPI_Select(void)
{
    HAL_GPIO_WritePin(SD_CS_GPIO_Port, SD_CS_Pin, GPIO_PIN_RESET);
    SD_SPI_TxRx(0xFF);
}

void SD_SPI_Unselect(void)
{
    HAL_GPIO_WritePin(SD_CS_GPIO_Port, SD_CS_Pin, GPIO_PIN_SET);
    SD_SPI_TxRx(0xFF);
}

uint8_t SD_SPI_TxRx(uint8_t data)
{
    uint8_t rx;
    HAL_SPI_TransmitReceive(&hspi2, &data, &rx, 1, HAL_MAX_DELAY);
    return rx;
}

void SD_SPI_SendDummyClocks(void)
{
    SD_SPI_Unselect();

    for (int i = 0; i < 10; i++)
    {
        SD_SPI_TxRx(0xFF);
    }
}
