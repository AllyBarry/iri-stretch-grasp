# Speech-to-text result for one utterance.
std_msgs/Header header
string utterance_id
string text
string language
float32 confidence        # 0..1, from ASR token log-probs (1.0 for typed input)
float32 no_speech_prob    # 0..1, ASR's belief the audio contained no speech
float32 audio_duration    # seconds
string[] uncertain_words  # words the recogniser had low confidence in (possible mis-hearings)
