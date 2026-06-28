/*
 * sdcard.c
 *
 *  Created on: Jun 27, 2026
 *      Author: hasmi
 */


#include "sdcard.h"
#include "sd_spi.h"

static uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc);
static uint8_t SD_Test_CMD0(void);
static uint8_t SD_Test_CMD8(uint8_t *r7);
static uint8_t SD_Init_ACMD41(void);
static uint8_t SD_ReadOCR(uint8_t *ocr);

static uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc)
{
    uint8_t response;

    SD_SPI_TxRx(0x40 | cmd);
    SD_SPI_TxRx((uint8_t)(arg >> 24));
    SD_SPI_TxRx((uint8_t)(arg >> 16));
    SD_SPI_TxRx((uint8_t)(arg >> 8));
    SD_SPI_TxRx((uint8_t)arg);
    SD_SPI_TxRx(crc);

    for (int i = 0; i < 100; i++)
    {
        response = SD_SPI_TxRx(0xFF);

        if ((response & 0x80) == 0)
        {
            return response;
        }
    }

    return 0xFF;
}

static uint8_t SD_Test_CMD0(void)
{
	SD_SPI_Init();

	SD_SPI_Select();

	uint8_t r = SD_SendCmd(0, 0x00000000, 0x95);

	SD_SPI_Unselect();

	return r;
}

static uint8_t SD_Test_CMD8(uint8_t *r7)
{
    uint8_t r;

    SD_SPI_Select();

    r = SD_SendCmd(8, 0x000001AA, 0x87);

    r7[0] = SD_SPI_TxRx(0xFF);
    r7[1] = SD_SPI_TxRx(0xFF);
    r7[2] = SD_SPI_TxRx(0xFF);
    r7[3] = SD_SPI_TxRx(0xFF);

    SD_SPI_Unselect();

    return r;
}

static uint8_t SD_Init_ACMD41(void)
{
    uint8_t r;
    uint32_t timeout = 10000;

    do
    {
        SD_SPI_Select();
        r = SD_SendCmd(55, 0x00000000, 0x01);
        SD_SPI_Unselect();

        if (r > 0x01)
        {
            return r;
        }

        SD_SPI_Select();
        r = SD_SendCmd(41, 0x40000000, 0x01);
        SD_SPI_Unselect();

        timeout--;

    } while (r != 0x00 && timeout > 0);

    return r;
}

static uint8_t SD_ReadOCR(uint8_t *ocr)
{
    uint8_t r;

    SD_SPI_Select();

    r = SD_SendCmd(58, 0x00000000, 0x01);

    ocr[0] = SD_SPI_TxRx(0xFF);
    ocr[1] = SD_SPI_TxRx(0xFF);
    ocr[2] = SD_SPI_TxRx(0xFF);
    ocr[3] = SD_SPI_TxRx(0xFF);

    SD_SPI_Unselect();

    return r;
}


uint8_t SD_Init(void)
{
    uint8_t r7[4];
    uint8_t ocr[4];

    uint8_t cmd0 = SD_Test_CMD0();

    if (cmd0 != 0x01)
    {
        return SD_ERROR;
    }

    uint8_t cmd8 = SD_Test_CMD8(r7);

    if (cmd8 != 0x01 || r7[2] != 0x01 || r7[3] != 0xAA)
    {
        return SD_ERROR;
    }

    uint8_t acmd41 = SD_Init_ACMD41();

    if (acmd41 != 0x00)
    {
        return SD_ERROR;
    }

    uint8_t cmd58 = SD_ReadOCR(ocr);

    if (cmd58 != 0x00)
    {
        return SD_ERROR;
    }

    return SD_OK;
}
uint8_t SD_ReadSector(uint32_t sector, uint8_t *buffer)
{
    uint8_t r;
    uint16_t timeout;

    SD_SPI_Select();

    r = SD_SendCmd(17, sector, 0x01);

    if (r != 0x00)
    {
        SD_SPI_Unselect();
        return SD_ERROR;
    }

    timeout = 0xFFFF;

    while (SD_SPI_TxRx(0xFF) != 0xFE)
    {
        if (--timeout == 0)
        {
            SD_SPI_Unselect();
            return SD_ERROR;
        }
    }

    for (uint16_t i = 0; i < 512; i++)
    {
        buffer[i] = SD_SPI_TxRx(0xFF);
    }

    // discard CRC
    SD_SPI_TxRx(0xFF);
    SD_SPI_TxRx(0xFF);

    SD_SPI_Unselect();

    return SD_OK;
}
uint8_t SD_WriteSector(uint32_t sector, const uint8_t *buffer)
{
    uint8_t r;
    uint8_t response;
    uint32_t timeout;

    SD_SPI_Select();

    r = SD_SendCmd(24, sector, 0x01);

    if (r != 0x00)
    {
        SD_SPI_Unselect();
        return SD_ERROR;
    }

    SD_SPI_TxRx(0xFF);
    SD_SPI_TxRx(0xFE);

    for (uint16_t i = 0; i < 512; i++)
    {
        SD_SPI_TxRx(buffer[i]);
    }

    SD_SPI_TxRx(0xFF);
    SD_SPI_TxRx(0xFF);

    response = SD_SPI_TxRx(0xFF);

    if ((response & 0x1F) != 0x05)
    {
        SD_SPI_Unselect();
        return SD_ERROR;
    }

    timeout = 0xFFFFF;

    while (SD_SPI_TxRx(0xFF) == 0x00)
    {
        if (--timeout == 0)
        {
            SD_SPI_Unselect();
            return SD_ERROR;
        }
    }

    SD_SPI_Unselect();

    return SD_OK;
}
