/*
 * wav_writer.h
 *
 *  Created on: Jun 28, 2026
 *      Author: hasmi
 */

#ifndef INC_WAV_WRITER_H_
#define INC_WAV_WRITER_H_


FRESULT WAV_CreateEmptyFile(const char *path,
                            uint32_t sample_rate,
                            uint16_t bits_per_sample,
                            uint16_t num_channels);

#endif /* INC_WAV_WRITER_H_ */
