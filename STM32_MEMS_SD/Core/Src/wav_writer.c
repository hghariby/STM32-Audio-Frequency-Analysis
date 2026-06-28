/*
 * wav_writer.c
 *
 *  Created on: Jun 28, 2026
 *      Author: hasmi
 */


#ifndef WAV_WRITER_H
#define WAV_WRITER_H

#include "fatfs.h"
#include <stdint.h>
#include "wav_writer.h"
#include <string.h>

static void write_u16_le(uint8_t *buf, uint16_t value)
{
    buf[0] = (uint8_t)(value);
    buf[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t *buf, uint32_t value)
{
    buf[0] = (uint8_t)(value);
    buf[1] = (uint8_t)(value >> 8);
    buf[2] = (uint8_t)(value >> 16);
    buf[3] = (uint8_t)(value >> 24);
}

FRESULT WAV_CreateEmptyFile(const char *path,
                            uint32_t sample_rate,
                            uint16_t bits_per_sample,
                            uint16_t num_channels)
{
    FIL file;
    FRESULT res;
    UINT bw;
    uint8_t header[44];

    uint32_t data_size = 0;
    uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    uint16_t block_align = num_channels * (bits_per_sample / 8);

    memset(header, 0, sizeof(header));

    memcpy(&header[0], "RIFF", 4);
    write_u32_le(&header[4], 36 + data_size);
    memcpy(&header[8], "WAVE", 4);

    memcpy(&header[12], "fmt ", 4);
    write_u32_le(&header[16], 16);
    write_u16_le(&header[20], 1);
    write_u16_le(&header[22], num_channels);
    write_u32_le(&header[24], sample_rate);
    write_u32_le(&header[28], byte_rate);
    write_u16_le(&header[32], block_align);
    write_u16_le(&header[34], bits_per_sample);

    memcpy(&header[36], "data", 4);
    write_u32_le(&header[40], data_size);

    res = f_open(&file, path, FA_CREATE_ALWAYS | FA_WRITE);
    if (res != FR_OK)
        return res;

    res = f_write(&file, header, sizeof(header), &bw);
    if (res == FR_OK && bw != sizeof(header))
        res = FR_DISK_ERR;

    f_close(&file);

    return res;
}
#endif
