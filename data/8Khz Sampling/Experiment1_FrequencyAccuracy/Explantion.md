This experiment was designed to evaluate the performance of Adafruit I2S MEMS Microphone Breakout (SPH0645LM4Haudio) in audio acquisition system used for frequency detection and signal analysis. The system records audio through the MEMS microphone connected to the STM32, and the captured signal is processed to determine how accurately different sound frequencies can be measured under varying conditions.

A key part of the system design comes from the way the microphone data is handled inside the microcontroller. The microphone produces a high-resolution digital signal, which is stored in a buffer as raw values. Before sending or analyzing this data, a bit-shifting operation is applied:

```c
for (int i = 0; i < 512; i++)
{
    pcm16[i] = (int16_t)((int32_t)micBuffer[i] >> 14);
}
```
The bit-shifting is done because the raw data coming from the MEMS microphone is not yet in a usable "audio signal format" for normal processing, and it is also too large in magnitude for the way the STM32 code later stores and transmits it.

Inside the STM32, the microphone (SPH0645) does not directly output simple 16-bit audio samples. Instead, it produces higher-bit internal digital signal. This raw value is stored in a 32-bit variable (micBuffer), and most of the meaningful audio information is contained in the upper bits of that number.

When this raw signal is used directly, two problems appear. First, the values are too large and do not fit cleanly into the standard 16-bit audio format used by WAV files and Python audio tools. Second, the signal includes extra resolution and noise in the lower bits that is not useful for frequency analysis and can actually make the signal less stable.

The bit-shifting operation solves both issues by scaling the signal down into a standard audio range. When the code shifts by 13, 14, or 15 bits, it is essentially dividing the signa. This reduces the amplitude so that it fits within the range of a signed 16-bit integer, which is the standard format for PCM audio.

Different STM32 audio examples found online use different scaling values, commonly 13, 14, or 15. 
Choosing 13, 14, or 15 changes how strong or weak the final signal appears, which can affect how clearly frequencies are detected later in processing. A smaller shift (like 13) keeps the signal stronger but risks distortion or clipping, while a larger shift (like 15) reduces amplitude and can make the signal weaker and more sensitive to noise. Since different examples and libraries online use different values depending on microphone gain, board configuration, and filtering approach, there is no universal correct shift value.
This step reduces the resolution of the raw microphone signal. Because there is no single universal correct value, all three versions were tested in this experiment to determine which one produces the cleanest and most reliable results for this specific setup.

The experimental process was structured in two main stages. In the first stage, audio recordings were collected at a fixed distance while gradually changing the input frequency. After that, the same set of frequencies was recorded again at increasing distances. This was done to observe how sound quality and different frequency detection change as the signal becomes weaker with distance. The same process was repeated for all three bit-shift configurations (13, 14, and 15) to compare their performance under identical conditions. The goal of this stage was to determine which configuration provides the most stable and accurate frequency detection across different distances and sound levels.

In the second stage, all collected recordings were analyzed using Python scripts. Each audio file was processed to extract its dominant frequency using FFT-based analysis. The detected frequency was then compared with the expected input frequency to measure accuracy. Additional analysis was performed to evaluate how clean and stable the recordings were. This included checking signal strength and consistency across different distances and configurations.

Through this analysis, it became possible to evaluate both the correctness of the recordings and the overall performance of the system. The results helped determine that an 8 kHz sampling rate is sufficient for detecting frequencies below approximately 4 kHz when using a 5-second recording window. It also provided insight into how different bit-shift values (13, 14, and 15) affect signal clarity and frequency detection reliability, allowing selection of the most suitable configuration for this specific hardware setup.
