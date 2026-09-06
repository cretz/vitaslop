//! Decoder configuration parsed from the 4-byte ATRAC9 config word (LibAtrac9
//! `decinit.c` `ReadConfigData` / `InitConfigData`).

use crate::bit_reader::BitReader;
use crate::tables::{
    ChannelConfig, CHANNEL_CONFIGS, SAMPLE_RATES, SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER,
};
use crate::Error;

/// Everything the decoder derives from the config word: rates, channel layout, and
/// frame/superframe geometry.
#[derive(Clone)]
pub(crate) struct Config {
    pub config_data: [u8; 4],
    pub sample_rate_index: i32,
    pub channel_config_index: i32,
    // Kept for fidelity with the reference config layout even though the decode
    // path derives everything it needs from the fields below.
    #[allow(dead_code)]
    pub frame_bytes: i32,
    #[allow(dead_code)]
    pub superframe_index: i32,

    pub channel_config: ChannelConfig,
    pub channel_count: i32,
    pub sample_rate: i32,
    pub high_sample_rate: bool,
    pub frames_per_superframe: i32,
    pub frame_samples_power: i32,
    pub frame_samples: i32,
    pub superframe_bytes: i32,
    #[allow(dead_code)]
    pub superframe_samples: i32,
}

impl Config {
    pub(crate) fn parse(config_data: [u8; 4]) -> Result<Config, Error> {
        let mut br = BitReader::new(&config_data);
        let header = br.read_int(8);
        let sample_rate_index = br.read_int(4);
        let channel_config_index = br.read_int(3);
        let validation_bit = br.read_int(1);
        let frame_bytes = br.read_int(11) + 1;
        let superframe_index = br.read_int(2);

        if header != 0xFE || validation_bit != 0 {
            return Err(Error::BadConfigData);
        }

        let frames_per_superframe = 1 << superframe_index;
        let superframe_bytes = frame_bytes << superframe_index;
        let channel_config = CHANNEL_CONFIGS[channel_config_index as usize];
        let channel_count = channel_config.channel_count as i32;
        let sample_rate = SAMPLE_RATES[sample_rate_index as usize];
        let high_sample_rate = sample_rate_index > 7;
        let frame_samples_power =
            SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER[sample_rate_index as usize];
        let frame_samples = 1 << frame_samples_power;
        let superframe_samples = frame_samples * frames_per_superframe;

        Ok(Config {
            config_data,
            sample_rate_index,
            channel_config_index,
            frame_bytes,
            superframe_index,
            channel_config,
            channel_count,
            sample_rate,
            high_sample_rate,
            frames_per_superframe,
            frame_samples_power,
            frame_samples,
            superframe_bytes,
            superframe_samples,
        })
    }
}
