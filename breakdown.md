# Nada agent

> An explanatory document which will take you through the inner workings of the system, including setup, architecture, self-report and my conclusion about our STS system.

---

## Contents:

- High-level overview
- Installation, setup and usage
- Imports and Agent-config class
- LLM Worker class (Ultravox)
- TTS Worker class (Kokoro)
- Audio Input Worker class
- Discussion and conclusion

---

## High-level Overview:

The system I have developed is a fusion of various strategies I have found to be useful through my research of many existing, state-of-the-art systems.

The system relies on the Ultravox llama 3.2 1B model, combining the STT and LLM stages, as well as Kokoro 82M for the TTS model. These lightweight models have allowed me to produce an interestingly capable system, easily running on my local device's GPU which hosts an RTX 3050 with 6GB VRAM.

In terms of architectural design, I will outline some key strategies which have been implemented in our system to optimise performance:

### Parallel Workers

Rather than having a sequential approach, I utilised parallel processing to significantly reduce latency. Our three independent workers (LLM Worker, TTS Worker, AudioInput Worker) run on separate threads simultaneously. This kills dead-time, as independent processes can occur without waiting for another process to end.

A key benefit of this is that while our TTS worker is outputting audio, our LLM worker can simultaneously process the input audio on a separate thread and produce more text output for the TTS worker to convert to audio.

Furthermore, since our AudioInput worker is always running concurrently, it is always ready to take in audio input, and this is what allows our instant interruption to work! It also prevents us having to dynamically switch threads which would introduce latency costs - concurreny saves us. This would not be possible in sequential architecture.

We initiate the parallel processes in the `if name == __main__` block.
The inner workings of each worker will be explained later in more detail.

### Speculative Execution:

This strategy seeks to combat the most critical delay of any speech-to-speech system: the time taken for the LLM to "think".

Typically, a naive approach would be to wait for 800ms or so to confirm that the user has stopped speaking (so that the agent doesn't mistakenly interrupt the user), after which the text output of the LLM would have to be processed into audio. This means that after the end of user speech, we would face about 800ms of dead-time, plus the time-to-first-token for Kokoro to produce the first audio token.

By speculative execution strategies, we only wait for 250ms of silence from the user to begin processing. Our system makes a bet to prioritise latency. It presumes on a 250ms pause that the user has finished speaking, and hence the AudioInput worker will send that audio snippet to the ultravox model and then the Kokoro model - i.e the audio snippet is sent to be fully processed through the pipeline. We then have a check at 600ms - has the user remained silent? If yes, then we can successfully output the produced speech and we would have won a huge latency gain. If the user did resume speaking however, then the remainder of the user speech will be discarded, and only the initial, incomplete audio snippet will be processed. In terms of user experience, this means that they receive quick responses, and in the case that we prematurely processed their speech, the agent would likely ask clarifying questions, or the human would clarify their intent in the next exchange.

Here is an example flow (next page):

> Let's use the example sentence: "Hey, what's the weather... [250ms pause] ...in Preston today?"
>
> 1.  The user says, "Hey, what's the weather," and pauses. The AudioInputWorker's timer starts. After 250ms of silence, the speculative condition is met.
> 2.  `self.speculative_trigger_time` is set to the current time. This is a critical flag that essentially says, "I have already sent a speculative request for this turn."
> 3.  A copy of the current `audio_buffer` (containing "Hey, what's the weather") is put onto the `llm_worker.input_queue`.
> 4.  **The Race Begins:** The LLMWorker, which is always waiting for work, immediately picks up this incomplete audio chunk and starts its time-consuming process: audio pre-processing, TTFT, etc. It is now trying to answer the question, "What's the weather?".
> 5.  **The User Resumes Speaking:** While the LLM is thinking, the user continues their sentence: "...in Preston today?".
> 6.  The AudioInputWorker's VAD detects this new speech (`is_speaking` becomes `True`).
> 7.  The code correctly appends the new audio chunks for "...in Preston today?" to its internal `audio_buffer`. The buffer now contains the full sentence.
> 8.  **The Inaction and the "Lost" Audio:** The user finally finishes speaking and falls silent.
> 9.  The AudioInputWorker now waits for the `END_OF_TURN_SILENCE_S` (0.6 seconds).
> 10. If condition met, it checks if it should send a request. The relevant line is: `if self.speculative_trigger_time is None and self.audio_buffer:`.
> 11. But because `self.speculative_trigger_time` was set back in step 2, this condition is `FALSE`. The code has no mechanism to send a second, corrected audio snippet.
> 12. The `_reset_state()` function is called, which clears the `audio_buffer` (deleting the full sentence) and resets `speculative_trigger_time` to `None`, preparing for the next turn.
>
> Hence, the agent may respond with something like "Where do you want to find out the weather for?"
>
> My personal thoughts on this will be discussed later.

### Instant "barge-in" interruption:

While the agent is speaking, we set `tts_worker.is_speaking` to `True`. Remember though, the AudioInputWorker is still actively listening for input speech. When input sound crosses the `VAD_INTERRUPT_CONFIDENCE` value, we call the interrupt function on the TTS Worker, and set the interruption event across all threads. All pending audio from the TTS Worker queues are removed, and the audio stream is terminated.

### Seamless audio output:

To mostly prevent gaps between speech output tokens, we use queues effectively.
`TTSWorker._main_tts_thread` simply takes the text output of the LLM, groups it into small logical phrases, feeds these to `Kokoro` to produce audio chunks and then has to `put()` these audio chunks onto the `audio_playback_queue`.
`TTS._audio_callback` uses `sounddevice` to spawn a parallel, high-priority thread to run this callback for audio chunks on the queue. This makes sure our audio chunks are played rapidly, one after the other, as the callback requests _one buffer's worth of audio at a time_. It is re-explainedin point 3 - audio output streaming.

### End-to-End Streaming Pipeline

Data is streamed through the pipeline as such:

> 1. LLM Output Streaming: The `LLMWorker` uses `transformers.TextIteratorStreamer`. This allows it to send out tokens of text as generated, rather than waiting for the full LLM response to be generated.

> 2. TTS Input Streaming and Natural Phrase Chunking: The `TTSWorker._main_tts_thread` consumes this token stream. To prevent robotic, word-by-word speech, it employs a buffering mechanism. Tokens from the LLM are accumulated in a token_buffer variable. A synthesis-and-dispatch action is only triggered when a complete, logical phrase is formed. This is determined by the following conditions within the `for token in iterator:` loop of the `TTSWorker._main_tts_thread` method:

- The buffer contains a "hard stop" punctuation mark ( , . ? ! )

- The buffer contains a comma and has grown beyond a certain word count (12 words).

- The buffer has exceeded a maximum word count (20 words), acting as a fallback for long, unpunctuated sentences.

By grouping tokens into these coherent chunks before calling \_generate_and_queue_audio, the system ensures the TTS engine has sufficient context to generate speech with natural-sounding intonation and rhythm (prosody).

> 3. Audio Output Streaming: The `TTSWorker._audio_callback` function plays back audio in small, continuous chunks from a queue, ensuring a seamless stream of sound to the speakers. This function is run in a separate, high-priority thread managed by the `sounddevice` library, with its role simply to `get()` audio chunks from `audio_playback_queue` and feed them to the hardware. This helps us meet the strict timing requirements of the audio hardware and prevent glitches, as mentioned earlier.

### Model Quantisation:

I have used 8-bit quantisation of the `transformers` library through `BitsAndBytesConfig(load_in_8bit = true)`- this loads model weights in 8bit integers rather than standard 16bit or 32bit floats. This allowed me to increase the efficiency of the system, allowing my hardware to run the model and speed up data transfer between GPU and memory, tackling a bottleneck in inference.

### Model Warm-ups:

By warming up models before any conversation, we address the "cold-start" issue, which is when the first conversational exchange has higher-than-expected latency due to the overhead of loading large models into memory. Through the `_warmup()` methods in `LLMWorker` and `TTSWorker` classes, we simulate a conversational exchange before the real user exchange, making sure models are loaded into memory beforehand so that the first real conversational exchange doesn't suffer from latency issues.

These are all the key architectural features outlined to you. We will now dive deeper into a complete breakdown of the code:

## Structured Code Breakdown

I will deliver a class-by-class, function-by-function breakdown of the code.

### `AgentConfig` and initialisation:

imported torch, numpy as required ML libraries
imported sounddevice to capture audio input and interact with audio hardware
imported sys for better logs
imported time for latency tracking in-development
imported os just to suppress warning logs

from transformers, we have imported AutoModel to download and load pre-trained ultravox model, AutoProcessor to load correct data processor for model, AutoConfig to load model's configuration file, TextIteratorStreamer to allow us to stream text to TTS, BitsAndBytesConfig for 8bit model quantisation.

We set `SAMPLE_RATE` to 16000Hz as standard, we set `LM_MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b`, we set `TTS_MODEL = 'hexgrad/Kokoro-82M`, and we set `VAD_MODEL_REPO = 'snakers4/silero-vad` for the automatic speech detection.

`VAD_SPEECH_CONFIDENCE` used as a threshold for whether or not the incoming audio contains human speech.
`VAD_INTERRUPT_CONFIDENCE` used as a threshold for whether the interruption audio contains human speech - kept higher than `VAD_SPEECH_CONFIDENCE` to reduce chance of the system registering stray noise or coughs as an interruption.

Note, theese thresholds are between 0 and 1, and are based on the probability measure obtained from the in-built function in `SileroVAD` which returns a probability value of audio containing human speech.

`SPECULATIVE_SILENCE_S = 0.25` used as threshold of 250ms of silence for our speculative execution strategy - upon 250ms of silence, the audio chunk will be cut and passed off to LLM Worker

`END_OF_TURN_SILENCE_S = 0.6` used as threshold of 600ms of silence to confirm user has stopped speaking. Hence, audio before the 250ms break will be packaged and sent to LLM Worker, and the state of the `AudioInputWorker` is reset using `self.reset_state()` (we will visit methods later in more detail). Any speech after this point will be classified as the start of a _new turn_. Note, we can modify the 600ms mark as appropriate to provide more flexibility - we'd just be trading latency for user experience.

`MIN and MAX speech speed` - allow for dynamic agent speech speed. Calculated based on how long the user pauses during speech to match their style.

`trim_silence()` finds all non-silent regions, then trims leading and trailing silent regions.

### AudioInputWorker class:

`__init__(self, config: AgentConfig, llm_input_queue: queue.Queue, tts_worker: TTSWorker)`:
takes `AgentConfig`, takes `llm_input_queue` as `Queue.queue` to send the audio output on a queue which is then taken by the LLM.
method also takes `tts_worker` reference to check for interruptions during the TTS worker's speaking state.
.
`self.vad_model` uses torch.hub.load to load `silero_vad` model without `onnx`.
`self.audio_buffer` set to empty array (no audio chunks in the buffer) and `self.state` set to "Listening".
Set silence start times to None - speculative execution not ready yet.

`start(self)`: launches `_run` method of AudioInputWorker in a new thread, with `daemon = True` to make sure the thread is "non-blocking".

`_run(self)`:
opens continuous stream using `sounddevice`, at `blocksize = 512` with float32 precision.
while loop checks for main global shutdown flag
we read an audio chunk of 512 frames from the daemon stream as a numpy array, flatten it to a 1d array, then convert it to a tensor to pass properly to silerovad, and then use `.item()` to retrieve the probability from silerovad that our audio chunk contains human speech. -
we then use this `speech_prob` to check for interruptions - check if TTS "is_speaking" and that speech_prob exceeds interruption confidence threshold, which follows with basic interruption logic. -
we then set the `is_speaking` variable based on whether it passes speech_confidence. If is_speaking, then we adjust state and immediately append the audio_chunk (512 frames) to the audio_buffer. We will then keep appending audio_chunks to this buffer during the user's speech. -
The state `listening` refers to when system is waiting to detect audio. The state `capturing` refers to when the system is actively receiving user's speech. -
If the system is `listening` and the user `is_speaking`, then reset state and transition to `capturing` state and appending captured `audio_chunk` to `audio_buffer`
If the system is already `capturing`, then it checks if user `is_speaking`. If yes, then it resets silence counter to 0, and appends another `audio_chunk`. If not, then we enter our speculative execution function. We first check if the user has been silent for more than 250ms AND we haven't hit silence yet (since `speculative_trigger_time` is None), then we send an object of metadata to LLM worker. -
The object contains: - `audio` - we join all `audio_chunks` in `audio_buffer` into a single 1d array and make a defensive copy of it too - `pause_duration` - experimented with 100ms pause to assess fallback measures and wait for perhaps resumation of speech - `trigger_time` - recorded speculative execution trigger time. -
We perform a similar process for non-speculative execution. -
`_reset_state(self)` - simple method to clear `audio_buffer` and set speculative execution trigger time and silence start time to None.

### LLM Worker class:

`__init__(self, config: AgentConfig, tts_worker: TTSWorker)`:
again takes the shared `AgentConfig`, and takes correct TTS worker, and initiates input queue
we load pretrained model, again with 8bit quantisation, using basic, reference-style attention mechanism "eager".
we also give the model a basic system prompt and config, and run a warmup.

`_warmup(self)`:
Basic warmup

`start(self)` : same as before

`_run(self)`:
Checks constantly for main global shutdown flag
.
Try block:
Fetch `item` - an object containing metadata produced by our `_run()` method in `AudioInputWorker`, taken from `input_queue`. If none, break
From metadata, we unpack raw audio, user's pause duration, and when VAD was triggered. Plus some latency logging.
.
Calculate `speed_factor` as a scalar to adjust `dynamic_speed` for audio playback speed.
Add basic system prompt with `<|audio|>` token and format properly with `tokenizer`. Importantly, here we are also preparing to feed conversation history as `self.conversation_turns` is an object containing all conversational exchanges so far.
We set `inputs` for pre-processing, taking audio+text pair, and send in the full conversational history with the most recent exchange, and wrap it all in the tensor required format. We then send it to the GPU for inference and log processing latency cost.
.
We initialise `streamer` as a generator to yield LLM output tokens as produced, and create empty array `full_response_text_list` ready to receive the full LLM response.
.
We assemble generation parameters in a dict, like max_tokens and sampling behaviour. We unpack the inputs (which include the full convo history as text prompt + latest audio buffer as audio prompt) and then use this to generate up to 150 new tokens. We also provide a `streamer`, so the LLM worker can use this to stream outputs. We create a safety copy by `do_sample`.
.
We send a `tts_payload` bundle to the TTS Worker, including our dynamic speed from before. The iterator for it is provided by `text_capturing_iterator(streamer)`.
Our `text_capturing_iterator(streamer)` function looks at the streamer we gave in generation kwargs, and yields tokens one by one from this streamer, and appends them to `full_response_text_list`. We also capture time-to-first-token. At this stage, we aim to produce tokens as fast as possible - grammatical correctness is handled by our TTS worker. We update the `tts_payload` iterator as such.
.
Importantly, note that our `text_capturing_iterator` is a lazy generator - it will not start producing all our LLM tokens and inference when we call `tts_payload['iterator'] = text_capturing_iterator(streamer)`, but rather this declaration will only pass the generator object to it. Our logic of the function will only start execution when we iterate over `tts_payload`, which we do soon with the `play()` method.
.
In the background, we spin off an `llm_thread`, and we launch the `generate` function for it and start it. The `generate` function comes from our pretrained model, and is what will produce our new tokens after being fed our generation kwargs. Remember, we also gave a `streamer` to the gen_kwargs, and so our generate function will now produce tokens and stream them. We can now run the TTS_worker's play method on our tts_payload, and so now we handoff tasks to our TTS worker.
.
We now just update our conversation history :)
.
except block: trivial

### TTS Worker class:

`__init__(self, config: AgentConfig, interruption_event: Event)`:
trivial setup, again taking shared `AgentConfig`, loading TTS inference pipeline as KPipeline.
Set llm_stream_queue as the output queue of our LLM worker, and so our TTS worker will get tokens from this queue.
Audio chunks synthesised by TTS will be stored in `audio_playback_queue`, and chunks immediately ready to be played by speakers are stored in `callback_queue`. Besides that, just some basic state management.

`start(self)`:
Same daemon thread initialisation as seen before. Notably, it shuts down with the main thread. This thread's only role is to continuously pull jobs from `llm_stream_queue`.

`_warmup(self)`:
Similar warmup, invoking generator function, running through the logic and inference to make sure models are properly loaded into memory, no output.

`play(self, item)`:
Takes `tts_payload` objects and puts them on `llm_stream_queue`, to decouple LLM and TTS processes, allowing asynchronous processing. Its role is for job control, not for audio processing. It just allows us to create a non-blocking interaction between LLM worker and TTS worker, and triggers the main tts method.
`stop(self)`:
Sets main shutdown event and interrupt event.

`interrupt(self)`:
Basic interruption logic and clears any remaining playback.

`_audio_callback(self, outdata, frames, time_info, status)`:
outdata is sounddevice's required write buffer, taking slices to immediately play audio - takes just one chunk .
`frames` is the number of samples requested, i.e the amount of room the `outdata` write buffer has. `time_info` is part of stream metadata. These two variables are required parameters of the callback interface defined by the sounddevice library. `status` also part of metadata.
.
We check if `callback_buffer` has more audio than `chunk_size`, if not, then we keep filling our callback_buffer.
Try:
We immediately try to grab an audio chunk from `audio_playback_queue`.
Then handling of "first" audio chunk, which may include log. We store the audio chunk as item, and display the log. For non-first audio, we simply store audio chunk as item.
If there is no audio chunk, then flush out any remaining sound in `callback_buffer` and close stream cleanly with callback.
Latency log for time-to-first-audio.
.
We now extend `callback_buffer` with fresh audio if it exists.
.
Except:
If no audio is in the `audio_playback_queue`, we measure the time it takes for audio to be received, and we give a speculative 50ms wait and try to grab audio again and add it to `callback_buffer` - if no audio is present still, then we play silence.
.
We construct outdata as a slice of our `callback_buffer` and remove that slice from the `callback_buffer`.

`_main_tts_thread(self)`:
first quick check for main global shutdown flag.
try:
Attempt to retrieve a `tts_payload` object from `llm_stream_queue` - if not then break. Make a quick latency log of this handoff.
We then extract `tts_payload` components as seen before, and then refresh state by clearing buffers and resetting events, ready to take in new data.
.
Set AI speaking state.
Set up a sounddevice output stream that repeatedly calls on our `_audio_callback` to play audio. (Note this reflects the role of play function - we're not taking audio from the play function's queue.)
We make sure to check for interruption event for instant halt to processes.
.
Again quick latency logs - monitor time between first processed llm token and first processed tts token.
We add tokens quickly to a `token_buffer`, and we split according to 3 criteria: hard stops, comma and more than 12 words, or more than 20 words.
We send off buffers to `_generate_and_queue_audio` if they meet any of the criteria, and reset the `token_buffer`.
We make sure to synthesise anything remaining in the buffers at the end of the stream.
.
Signal for process completion, wait until all audio is played, then close the thread.
.
except:
log error
.
finally:
Set state as Idle.

`_generate_and_queue_audio(self, text, speed)`:
Immediate interruption check, plus blank/whitespace check to exit.
.
try:
flush out text sent to TTS engine out to the terminal.
invoke generator function to emit audio chunks for given `text` from kokoro.KPipeline, and signals first chunk flag.
for each chunk, immediate interruption check.
if first chunk then basic latency logging (experimenting with "cold-start" effect)
convert audio tensor chunks to numpy arrays, trim leading/trailing silence and then add latency log.
check if audio chunk is first chunk (hence it will be tuple with extra latency logs), checks if it has non-zero values (not silence), then puts item onto `audio_playback_queue`.
handle any exceptions
