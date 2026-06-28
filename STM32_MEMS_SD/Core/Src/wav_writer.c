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

FRESULT WAV_CreateDummyToneFile(const char *path,
                                uint32_t sample_rate,
                                uint16_t bits_per_sample,
                                uint16_t num_channels,
                                uint32_t duration_seconds)
{
    FIL file;
    FRESULT res;
    UINT bw;

    uint8_t header[44];
    uint8_t sample[2] = {0x00, 0x00};      // 16-bit PCM silence

    uint32_t sample_count =
        sample_rate * duration_seconds * num_channels;

    uint32_t data_size = sample_count * 2;
    uint32_t byte_rate = sample_rate * num_channels * 2;
    uint16_t block_align = num_channels * 2;

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
    if (res != FR_OK)
    {
        f_close(&file);
        return res;
    }

    for (uint32_t i = 0; i < sample_count; i++)
    {
        res = f_write(&file, sample, 2, &bw);
        if (res != FR_OK)
            break;
    }

    f_close(&file);

    return res;
}

FRESULT WAV_Start(FIL *file,
                  const char *path,
                  uint32_t sample_rate,
                  uint16_t bits_per_sample,
                  uint16_t num_channels)
{
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

    res = f_open(file, path, FA_CREATE_ALWAYS | FA_WRITE);
    if (res != FR_OK)
        return res;

    res = f_write(file, header, sizeof(header), &bw);
    if (res != FR_OK || bw != sizeof(header))
    {
        f_close(file);
        return FR_DISK_ERR;
    }

    return FR_OK;
}

FRESULT WAV_WriteSamples(FIL *file,
                         const int16_t *samples,
                         uint32_t sample_count)
{
    UINT bw;
    uint32_t bytes_to_write = sample_count * sizeof(int16_t);

    FRESULT res = f_write(file, samples, bytes_to_write, &bw);

    if (res != FR_OK)
        return res;

    if (bw != bytes_to_write)
        return FR_DISK_ERR;

    return FR_OK;
}

FRESULT WAV_Finish(FIL *file)
{
    FRESULT res;
    UINT bw;
    FSIZE_t file_size;
    uint32_t riff_size;
    uint32_t data_size;
    uint8_t size_buf[4];

    file_size = f_size(file);

    if (file_size < 44)
        return FR_INT_ERR;

    data_size = (uint32_t)(file_size - 44);
    riff_size = 36 + data_size;

    write_u32_le(size_buf, riff_size);
    res = f_lseek(file, 4);
    if (res != FR_OK)
        return res;

    res = f_write(file, size_buf, 4, &bw);
    if (res != FR_OK || bw != 4)
        return FR_DISK_ERR;

    write_u32_le(size_buf, data_size);
    res = f_lseek(file, 40);
    if (res != FR_OK)
        return res;

    res = f_write(file, size_buf, 4, &bw);
    if (res != FR_OK || bw != 4)
        return FR_DISK_ERR;

    res = f_close(file);

    return res;
}
#endif
