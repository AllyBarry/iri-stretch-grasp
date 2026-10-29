# One voice-activity-segmented utterance (mono, signed 16-bit PCM).
std_msgs/Header header
string utterance_id
uint32 sample_rate
int16[] samples
