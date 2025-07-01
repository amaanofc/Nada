## Core Thread Architecture

This diagram shows which threads operate independently, what queues they watch or feed, and where parallelism enables low-latency performance.

```mermaid
    flowchart TD

    subgraph AudioInputWorker Thread
        A1["AudioInputWorker._run()"]
        A1 --> Q1["→ LLMWorker.input_queue.put(audio_chunk)"]
    end

    subgraph LLMWorker Thread
        B1["LLMWorker._run()"]
        B1 --> B2["model.generate(..., streamer) [Background Thread]"]
        B2 --> Q2["→ TTSWorker.play(tts_payload)"]
    end

    subgraph TTSWorker Thread
        C1["TTSWorker._main_tts_thread()"]
        C1 --> C2["_generate_and_queue_audio(text)"]
        C2 --> Q3["→ audio_playback_queue.put(chunk)"]
    end

    subgraph Audio Callback Thread
        D1["TTSWorker._audio_callback(outdata, ...)"]
        D1 --> Speaker["sd.OutputStream (speaker thread)"]
    end
```

- AudioInputWorker.\_run() detects voice activity and places audio on input_queue
- LLMWorker.\_run() processes audio and spawns generate() in a background thread
- TTSWorker.play() receives tts_payload and queues it for speech synthesis
- TTSWorker.\_main_tts_thread() handles buffering, synthesis, and queuing playback
- sd.OutputStream calls \_audio_callback() to play buffered audio via callback_buffer

## Execution Pipeline: Audio to Speech

This flow diagram traces how audio is transformed into spoken output using streaming token generation and real-time synthesis.

```mermaid
    flowchart TD
        Mic["User Microphone (sd.InputStream)"] --> AudioIn["AudioInputWorker._run()"]
        AudioIn --> LLMQueue["LLMWorker.input_queue (Queue)"]
        LLMQueue --> LLM["LLMWorker._run()"]
        LLM --> Model["model.generate(..., streamer) [Background Thread]"]
        Model --> Streamer["TextIteratorStreamer"]
        Streamer --> Iterator["text_capturing_iterator(streamer)"]

        Iterator --> TTSQueue["TTSWorker.llm_stream_queue (Queue)"]
        TTSQueue --> TTSThread["TTSWorker._main_tts_thread()"]
        TTSThread --> PlaybackQueue["TTSWorker.audio_playback_queue (Queue)"]
        PlaybackQueue --> Callback["TTSWorker._audio_callback(outdata)"]
        Callback --> Speaker["System Audio (via sd.OutputStream)"]
        Speaker --> User["User hears response"]

        AudioIn -- interruption_event.set() --> TTSThread
```

### Key Code References

| Component | Code Reference | Role |
| AudioInputWorker | AudioInputWorker.\_run() | Detects speech, sends chunks |
| LLMWorker | LLMWorker.\_run() | Prepares tokens via model.generate(...) |
| TextIteratorStreamer | Wrapped in text_capturing_iterator | Streams tokens to TTS |
| llm_stream_queue | TTSWorker.llm_stream_queue | Signals new speaking turn |
| audio_playback_queue | TTSWorker.audio_playback_queue | Buffers synthesized audio |
| interruption_event | TTSWorker.interruption_event | Interrupts generation and playback |
| callback_buffer | TTSWorker.callback_buffer | Final staging area before playback |
| sd.OutputStream | Starts in \_main_tts_thread() | Pulls audio frames via callback |

## Class Responsibilities

The following class relationships highlight dependency injection, shared config, and queue references.

```mermaid
    classDiagram
    class AgentConfig {
        +LLM_MODEL
        +TTS_MODEL
        +VAD_MODEL_REPO
        +SAMPLE_RATE
        +END_OF_TURN_SILENCE_S
    }

    class AudioInputWorker {
        -output_queue: Queue
        -tts_worker: TTSWorker
        +start()
        -_run()
        -_reset_state()
    }

    class LLMWorker {
        -input_queue: Queue
        -tts_worker: TTSWorker
        -model
        +start()
        -_run()
        +play(item)
    }

    class TTSWorker {
        -llm_stream_queue: Queue
        -audio_playback_queue: Queue
        -interruption_event: Event
        +start()
        -_main_tts_thread()
        -_generate_and_queue_audio()
        -_audio_callback()
    }

    AudioInputWorker --> LLMWorker
    AudioInputWorker --> TTSWorker
    LLMWorker --> TTSWorker
    AudioInputWorker --|> AgentConfig
    LLMWorker --|> AgentConfig
    TTSWorker --|> AgentConfig
```

### Clarified Purpose

- AgentConfig provides global configuration
- All worker classes use queues for thread-safe data sharing
- TTSWorker is shared across the system to unify synthesis

- LLMWorker.input_queue and TTSWorker.llm_stream_queue are the two coordination points between perception and speech.
- AgentConfig enforces consistency across modules — sample rate, model paths, etc.
- Thread separation enables parallel voice response without blocking recognition.

## Sequence of Execution

This shows how each part activates and hands off responsibility. Useful for tracing performance or debugging latency.

```mermaid
    sequenceDiagram
    participant User
    participant VAD as AudioInputWorker._run()
    participant LLM as LLMWorker._run()
    participant TTS as TTSWorker._main_tts_thread()
    participant Callback as TTSWorker._audio_callback()

    User->>VAD: Starts speaking
    VAD->>VAD: Detects pause (> END_OF_TURN_SILENCE_S)
    VAD->>LLM: output_queue.put(audio_chunk)

    LLM->>LLM: input_queue.get()
    LLM->>LLM: Inference (self.model.generate(streamer))
    LLM->>TTS: play(tts_payload ➝ llm_stream_queue.put())

    TTS->>TTS: llm_stream_queue.get()
    TTS->>TTS: Iterate & buffer tokens
    TTS->>TTS: _generate_and_queue_audio(text)
    TTS->>Callback: audio_playback_queue.put(chunk)

    Callback->>Callback: get audio chunk
    Callback-->>User: Play via sd.OutputStream
```

### Precise Execution Points

| Stage | Code |
| VAD Decision | AudioInputWorker.\_run() triggers on silence |
| LLM Thread Start | Thread(target=generate, ...) |
| TTS Coordination | TTSWorker.play(...) ➝ Queue |
| Audio Synthesis | TTSWorker.\_generate_and_queue_audio() |
| Audio Playback | TTSWorker.\_audio_callback() |

## Voice Detection States

Shows how speculative vs committed turn transitions work.

```mermaid
    stateDiagram-v2
    [*] --> LISTENING
    LISTENING --> CAPTURING: Speech Detected
    CAPTURING --> CAPTURING: More speech
    CAPTURING --> LISTENING: Final silence (> END_OF_TURN_SILENCE_S)

    note right of CAPTURING
    After SPECULATIVE_SILENCE_S seconds,
    LLM inference starts in background,
    but playback waits for final silence.
    end note
```

## Streaming, Phrase-Flushing, and Interrupts

Streaming tokens directly into speech is central to responsiveness. Interrupts allow mid-sentence cancellation.

```mermaid
    flowchart TD
    Gen["LLMWorker: model.generate(..., streamer)"] --> Streamer["TextIteratorStreamer"]
    Streamer --> Iterator["text_capturing_iterator(streamer)"]
    Iterator --> ForLoop["for token in iterator"]
    ForLoop --> Buffer["token_buffer += token"]
    Buffer --> FlushCheck["if phrase boundary"]
    FlushCheck --> Synthesis["_generate_and_queue_audio(token_buffer)"]
    Synthesis --> PlaybackQueue["audio_playback_queue.put(chunk)"]

    User["User interrupts"] --> Interrupt["interruption_event.set()"]
    Interrupt --> Cancel["Playback and generation terminated"]
```

- Phrase flushing triggered via punctuation or token count
- First audio chunk timestamped to measure responsiveness
- Audio stream ends gracefully via audio_playback_queue.put(None)

## Voice Detection and Speculative Inference

This diagram shows how early inference starts before speech ends.

```mermaid
    stateDiagram-v2
    [*] --> LISTENING
    LISTENING --> CAPTURING : Speech Detected
    CAPTURING --> CAPTURING : More speech
    CAPTURING --> SPECULATING : Silence > SPECULATIVE_SILENCE_S

    note right of SPECULATING
    LLM inference has started,
    but playback is not yet triggered.
    end note

    SPECULATING --> LISTENING : Final silence > END_OF_TURN_SILENCE_S
    CAPTURING --> LISTENING : Interrupted or speech resumes`
```

### Relevant Code Anchors

| Variable | File | Role |
| SPECULATIVE_SILENCE_S | AgentConfig | Starts early token generation |
| END_OF_TURN_SILENCE_S | AgentConfig | Confirms true end of user speech |
| interruption_event | TTSWorker | Cancels synthesis and playback |
| first_audio_chunk_queued_time | \_generate_and_queue_audio() | Measures model → speech latency |

### Relevant Variables

| Variable | Purpose |
| AgentConfig.SPECULATIVE_SILENCE_S | Starts speculative generation |
| AgentConfig.END_OF_TURN_SILENCE_S | Confirms final turn handoff |
| AudioInputWorker.is_speaking | Checked by TTSWorker to decide if playback is allowed |
| interruption_event | Used to halt generation and audio synthesis during overlapping speech |

### Component Summary

| Component | Method / Object | Role |
| AudioInputWorker | \_run() | Detects voice, triggers LLM |
| LLMWorker | \_run() → generate() | Streams tokens |
| TTSWorker | play() → \_main_tts_thread() | Synthesizes speech |
| sd.OutputStream | Callback to \_audio_callback() | Delivers audio |
| text_capturing_iterator | Wraps token stream | Adds timing markers |
| audio_playback_queue | Queue | Buffers chunks for playback |
| callback_buffer | np.ndarray | Staging area for speaker feed |
