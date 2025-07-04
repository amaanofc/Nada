# main.py (V58 - Latency-Profiled Engine)

import torch
import numpy as np
import sounddevice as sd
import sys
import time
from transformers import AutoModel, AutoProcessor, AutoConfig, TextIteratorStreamer, BitsAndBytesConfig
import queue
from threading import Thread, Event
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AgentConfig:
    SAMPLE_RATE = 16000
    LLM_MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
    TTS_MODEL = 'hexgrad/Kokoro-82M'
    VAD_MODEL_REPO = 'snakers4/silero-vad'
    VAD_SPEECH_CONFIDENCE = 0.4
    VAD_INTERRUPT_CONFIDENCE = 0.75
    SPECULATIVE_SILENCE_S = 0.25
    END_OF_TURN_SILENCE_S = 0.6
    MIN_SPEECH_SPEED = 1.0
    MAX_SPEECH_SPEED = 1.25

def trim_silence(audio_chunk, threshold=0.01):
    non_silent_indices = np.where(np.abs(audio_chunk) > threshold)[0]
    if len(non_silent_indices) == 0: return audio_chunk
    return audio_chunk[non_silent_indices[0]:non_silent_indices[-1] + 1]

class TTSWorker:
    def __init__(self, config: AgentConfig, interruption_event: Event):
        from kokoro import KPipeline
        print("[TTS] Initializing...", file=sys.stderr)
        self.config = config
        self.interruption_event = interruption_event
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = KPipeline(repo_id=self.config.TTS_MODEL, lang_code='en-US', device=self.device)

        self.llm_stream_queue = queue.Queue()
        self.audio_playback_queue = queue.Queue(maxsize=200)
        self.callback_buffer = np.array([], dtype=np.float32)
        ### PERF MARKER ### - Renamed for clarity
        self.initial_trigger_time = None
        self.first_audio_chunk_queued_time = None

        self.is_speaking = False
        self.shutdown_event = Event()
        self.playback_complete_event = Event()

    def start(self):
        Thread(target=self._main_tts_thread, daemon=True).start()
        
    def _warmup(self):
        try:
            print("[TTS] Warming up engine with a full sentence...")
            # Use a more realistic sentence to ensure all components are initialized.
            warmup_text = "This is a warm-up sentence to prepare the text-to-speech engine for real-time synthesis."
            generator = self.pipeline(warmup_text, voice='af_heart', speed=1.2)
            # Iterate through the generator to complete the synthesis
            for _, _, _ in generator:
                pass
            print("[TTS] Warm-up complete.", file=sys.stderr)
        except Exception as e:
            print(f"[TTS] Warm-up failed: {e}", file=sys.stderr)
    
    def play(self, item): self.llm_stream_queue.put(item)
    def stop(self): self.shutdown_event.set(); self.interrupt()
    
    def interrupt(self):
        if self.is_speaking:
            print("\n[TTS] Playback and generation pipeline forcefully interrupted.", file=sys.stderr)
            self.interruption_event.set()
            self.playback_complete_event.set()
            while not self.audio_playback_queue.empty():
                try: self.audio_playback_queue.get_nowait()
                except queue.Empty: continue

    def _audio_callback(self, outdata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        if self.interruption_event.is_set(): raise sd.CallbackStop

        chunk_size = len(outdata)
        
        while len(self.callback_buffer) < chunk_size:
            try:
                item = self.audio_playback_queue.get_nowait()
                if isinstance(item, tuple):
                    audio_chunk, _ = item
                    ### PERF MARKER ### - Capture when the first audio is received by the callback
                    if self.first_audio_chunk_queued_time:
                         print(f"[PERF] TTS Queue -> Audio Callback: {(time.perf_counter() - self.first_audio_chunk_queued_time) * 1000:.2f}ms", file=sys.stderr)
                         self.first_audio_chunk_queued_time = None # Only print once
                else:
                    audio_chunk = item

                if audio_chunk is None:
                    if len(self.callback_buffer) > 0:
                        outdata[:len(self.callback_buffer)] = self.callback_buffer.reshape(-1, 1); outdata[len(self.callback_buffer):] = 0
                        self.callback_buffer = np.array([], dtype=np.float32)
                    raise sd.CallbackStop
                
                ### PERF MARKER ### - The final, ground-truth measurement
                if self.initial_trigger_time:
                    print(f"[PERF] ========================================================", file=sys.stderr)
                    print(f"[PERF] TOTAL LATENCY (VAD Trigger to First Sound): {(time.perf_counter() - self.initial_trigger_time) * 1000:.2f}ms", file=sys.stderr)
                    print(f"[PERF] ========================================================", file=sys.stderr)
                    self.initial_trigger_time = None

                self.callback_buffer = np.concatenate((self.callback_buffer, audio_chunk))
            except queue.Empty:
                t_gap_start = time.perf_counter()
                try:
                    item = self.audio_playback_queue.get(timeout=0.05)
                    if isinstance(item, tuple): item = item[0] # Discard time for subsequent chunks
                    if item is None: raise sd.CallbackStop
                    self.callback_buffer = np.concatenate((self.callback_buffer, item))
                    print(f"[PERF] Audio Pipeline Gap (waited for data): {(time.perf_counter() - t_gap_start) * 1000:.2f}ms", file=sys.stderr)
                except queue.Empty: outdata.fill(0); return
        
        outdata[:] = self.callback_buffer[:chunk_size].reshape(-1, 1)
        self.callback_buffer = self.callback_buffer[chunk_size:]

    def _main_tts_thread(self):
        while not self.shutdown_event.is_set():
            try:
                item = self.llm_stream_queue.get(timeout=1)
                if item is None: break
                
                ### PERF MARKER ###
                t_tts_receives_item = time.perf_counter()
                print(f"[PERF] LLM -> TTS Queue Handoff: {(t_tts_receives_item - item['t_llm_handoff']) * 1000:.2f}ms", file=sys.stderr)

                iterator, dynamic_speed, self.initial_trigger_time = item['iterator'], item['speed'], item.get('initial_trigger_time')
                
                self.interruption_event.clear(); self.playback_complete_event.clear()
                self.callback_buffer = np.array([], dtype=np.float32)
                while not self.audio_playback_queue.empty(): self.audio_playback_queue.get_nowait()
                
                self.is_speaking = True
                print("[State] -> AI SPEAKING", file=sys.stderr)
                
                stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32',
                                         callback=self._audio_callback,
                                         finished_callback=self.playback_complete_event.set)
                stream.start()
                
                ### PERF MARKER ###
                t_tts_processing_first_token = None
                
                token_buffer = ""
                for token in iterator:
                    if self.interruption_event.is_set(): break
                    
                    ### PERF MARKER ###
                    if t_tts_processing_first_token is None:
                        t_tts_processing_first_token = time.perf_counter()
                        print(f"[PERF] LLM First Token -> TTS First Token Processing: {(t_tts_processing_first_token - item['t_first_token']) * 1000:.2f}ms", file=sys.stderr)

                    token_buffer += token
                    words, hard_stop_chars = token_buffer.split(), {'.', '?', '!', ':"'}
                    if any(c in token_buffer for c in hard_stop_chars) or (',' in token_buffer and len(words) > 12) or len(words) > 20:
                        self._generate_and_queue_audio(token_buffer, dynamic_speed)
                        token_buffer = ""
                
                if token_buffer.strip() and not self.interruption_event.is_set():
                    self._generate_and_queue_audio(token_buffer, dynamic_speed)
                
                self.audio_playback_queue.put(None)
                self.playback_complete_event.wait()
                stream.close()

            except queue.Empty: continue
            except Exception as e: print(f"TTS Main Thread Error: {e}", file=sys.stderr)
            finally:
                self.is_speaking = False
                print("[State] -> AI IDLE", file=sys.stderr)

    def _generate_and_queue_audio(self, text, speed):
        if not text.strip() or self.interruption_event.is_set(): return
        
        try:
            print(f"[TTS] > {text}", file=sys.stdout); sys.stdout.flush()
            t_start_tts = time.perf_counter()
            generator = self.pipeline(text, voice='af_heart', speed=speed)
            
            first_chunk = True
            for i, (_, _, audio_chunk) in enumerate(generator):
                if self.interruption_event.is_set(): return
                
                ### PERF MARKER ###
                if first_chunk:
                    print(f"[PERF] TTS First Chunk Synthesis Time: {(time.perf_counter() - t_start_tts) * 1000:.2f}ms", file=sys.stderr)
                    self.first_audio_chunk_queued_time = time.perf_counter()
                    # Pass the timestamp along with the first chunk
                    item_to_queue = (trim_silence(audio_chunk.cpu().numpy()), self.first_audio_chunk_queued_time)
                    first_chunk = False
                else:
                    item_to_queue = trim_silence(audio_chunk.cpu().numpy())

                if np.any(item_to_queue[0]) if isinstance(item_to_queue, tuple) else np.any(item_to_queue):
                     self.audio_playback_queue.put(item_to_queue)

        except Exception as e: print(f"[TTS] Generation error: {e}", file=sys.stderr)

class LLMWorker:
    def __init__(self, config: AgentConfig, tts_worker: TTSWorker):
        print("[LLM] Initializing...", file=sys.stderr)
        self.config, self.tts_worker, self.input_queue = config, tts_worker, queue.Queue()
        
        model_config = AutoConfig.from_pretrained(config.LLM_MODEL, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            config.LLM_MODEL, config=model_config, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_8bit=True), attn_implementation="eager"
        )
        self.processor = AutoProcessor.from_pretrained(config.LLM_MODEL, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        system_prompt = "You are a helpful, friendly, and eloquent conversational AI assistant. Keep your responses concise, natural, and accurate."
        self.conversation_turns = [{"role": "system", "content": system_prompt}]
        self._warmup()
        print("[LLM] Ready.", file=sys.stderr)

    def _warmup(self):
        try:
            print("[LLM] Warming up engine...")
            dummy_text = "User: <|audio>"
            dummy_audio = np.zeros(self.config.SAMPLE_RATE, dtype=np.float32)
            inputs = self.processor(text=dummy_text, audios=[dummy_audio], sampling_rate=self.config.SAMPLE_RATE, return_tensors="pt").to(self.model.device)
            _ = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            print("[LLM] Warm-up complete.", file=sys.stderr)
        except Exception as e: print(f"[LLM] Warm-up failed: {e}", file=sys.stderr)

    def start(self): Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not main_shutdown_event.is_set():
            try:
                item = self.input_queue.get(timeout=1)
                if item is None: break

                ### PERF MARKER ### - Renamed eot_time to be more generic
                audio_data, pause_duration, initial_trigger_time = item['audio'], item['pause_duration'], item['trigger_time']
                
                t_llm_start = time.perf_counter()
                print(f"\n[PERF] --------------------------------------------------------", file=sys.stderr)
                print(f"[PERF] VAD Trigger -> LLM Worker Start: {(t_llm_start - initial_trigger_time) * 1000:.2f}ms", file=sys.stderr)
                print("[State] -> AI PROCESSING", file=sys.stderr)
                
                speed_factor = 1 - min(max(pause_duration, 0.5), 1.5) / 1.0
                dynamic_speed = self.config.MIN_SPEECH_SPEED + (self.config.MAX_SPEECH_SPEED - self.config.MIN_SPEECH_SPEED) * speed_factor
                
                prompt_turns = self.conversation_turns + [{"role": "user", "content": "<|audio|>"}]
                text_prompt = self.tokenizer.apply_chat_template(prompt_turns, tokenize=False, add_generation_prompt=True)
                
                t_process_start = time.perf_counter()
                inputs = self.processor(text=text_prompt, audios=[audio_data], sampling_rate=self.config.SAMPLE_RATE, return_tensors="pt").to(self.model.device)
                t_process_end = time.perf_counter()
                print(f"[PERF] LLM Audio Pre-processing: {(t_process_end - t_process_start) * 1000:.2f}ms", file=sys.stderr)
                
                # --- Streaming Setup ---
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                full_response_text_list = []
                
                generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=150, do_sample=True, temperature=0.7)
                t_start_generate = time.perf_counter()

                # This dictionary will be passed to the TTS worker.
                # The iterator function below will modify it as a side effect.
                tts_payload = {
                    'iterator': None, # Will be set below
                    'speed': dynamic_speed, 
                    'initial_trigger_time': initial_trigger_time,
                    't_first_token': None, # This will be populated by the iterator
                    't_llm_handoff': time.perf_counter()
                }
                
                # This iterator captures tokens for the full response text
                # and updates the tts_payload with the first token's timestamp.
                def text_capturing_iterator(streamer_instance):
                    nonlocal full_response_text_list
                    for token in streamer_instance:
                        if tts_payload['t_first_token'] is None:
                            tts_payload['t_first_token'] = time.perf_counter()
                            print(f"[PERF] LLM Time To First Token (TTFT): {(tts_payload['t_first_token'] - t_start_generate) * 1000:.2f}ms", file=sys.stderr)
                        full_response_text_list.append(token)
                        yield token
                
                tts_payload['iterator'] = text_capturing_iterator(streamer)

                # --- Start Generation and Handoff to TTS ---
                llm_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                llm_thread.start()
                
                self.tts_worker.play(tts_payload)
                
                llm_thread.join()

                # --- Finalize Conversation History ---
                final_text = "".join(full_response_text_list).strip()
                self.conversation_turns.append({"role": "user", "content": "[user audio]"})
                if final_text: self.conversation_turns.append({"role": "assistant", "content": final_text})
                
            except queue.Empty: continue
            except Exception as e: print(f"[LLM Error] {e}", file=sys.stderr)

class AudioInputWorker:
    def __init__(self, config: AgentConfig, llm_input_queue: queue.Queue, tts_worker: TTSWorker):
        print("[VAD] Initializing...", file=sys.stderr)
        self.config, self.output_queue, self.tts_worker = config, llm_input_queue, tts_worker
        self.vad_model, _ = torch.hub.load(repo_or_dir=config.VAD_MODEL_REPO, model='silero_vad', onnx=False, trust_repo=True)
        self.audio_buffer, self.state = [], "LISTENING"
        self.silence_start_time, self.speculative_trigger_time = None, None
        print(f"[VAD] Ready.", file=sys.stderr)

    def start(self): Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        print("[State] -> AI LISTENING", file=sys.stderr)
        with sd.InputStream(samplerate=self.config.SAMPLE_RATE, channels=1, dtype='float32', blocksize=512) as stream:
            while not main_shutdown_event.is_set():
                try:
                    audio_chunk, _ = stream.read(stream.blocksize)
                    speech_prob = self.vad_model(torch.from_numpy(audio_chunk.flatten()), self.config.SAMPLE_RATE).item()
                    
                    if self.tts_worker.is_speaking and speech_prob > self.config.VAD_INTERRUPT_CONFIDENCE:
                        print("\n[VAD] Interruption detected!", file=sys.stderr)
                        self.tts_worker.interrupt()
                        self._reset_state()
                        self.state = "CAPTURING"
                        self.audio_buffer.append(audio_chunk)
                        continue

                    is_speaking = speech_prob > self.config.VAD_SPEECH_CONFIDENCE
                    
                    if self.state == "LISTENING":
                        if is_speaking: print("[State] -> USER SPEAKING", file=sys.stderr); self._reset_state(); self.state = "CAPTURING"; self.audio_buffer.append(audio_chunk)
                    elif self.state == "CAPTURING":
                        if is_speaking: self.silence_start_time = None; self.audio_buffer.append(audio_chunk)
                        else:
                            if self.silence_start_time is None: self.silence_start_time = time.perf_counter()
                            
                            current_silence_duration = time.perf_counter() - self.silence_start_time

                            if self.speculative_trigger_time is None and (current_silence_duration > self.config.SPECULATIVE_SILENCE_S):
                                print(f"[VAD] Speculative trigger after {current_silence_duration*1000:.2f}ms of silence.", file=sys.stderr)
                                self.speculative_trigger_time = time.perf_counter()
                                if self.audio_buffer: self.output_queue.put({'audio': np.concatenate(self.audio_buffer).flatten().copy(), 'pause_duration': 0.1, 'trigger_time': self.speculative_trigger_time})

                            if current_silence_duration > self.config.END_OF_TURN_SILENCE_S:
                                eot_time = time.perf_counter()
                                print(f"[VAD] End of turn confirmed after {current_silence_duration*1000:.2f}ms of silence.", file=sys.stderr)
                                if self.speculative_trigger_time is None and self.audio_buffer:
                                    # This is the non-speculative path
                                    self.output_queue.put({'audio': np.concatenate(self.audio_buffer).flatten(), 'pause_duration': current_silence_duration, 'trigger_time': eot_time})
                                self._reset_state()
                except Exception as e: print(f"Audio input error: {e}", file=sys.stderr); self._reset_state()

    def _reset_state(self):
        self.state = "LISTENING"; self.audio_buffer.clear()
        self.silence_start_time, self.speculative_trigger_time = None, None

if __name__ == "__main__":
    main_shutdown_event, config = Event(), AgentConfig()
    interruption_event = Event()
    tts_worker = TTSWorker(config, interruption_event)
    llm_worker = LLMWorker(config, tts_worker)
    audio_input = AudioInputWorker(config, llm_worker.input_queue, tts_worker)
    
    tts_worker.start(); llm_worker.start(); audio_input.start()
    time.sleep(2)
    # Perform warm-up after threads have started
    llm_worker._warmup()
    tts_worker._warmup()

    print("\n--- All Systems Ready ---"); print(">>> AGENT IS LISTENING. SPEAK NOW. (Ctrl+C to exit) <<<")

    try:
        while not main_shutdown_event.is_set(): time.sleep(1)
    except KeyboardInterrupt: print("\n[MAIN] Shutdown signal received.", file=sys.stderr)
    finally:
        main_shutdown_event.set(); tts_worker.stop()
        if llm_worker.input_queue.empty():
            try: llm_worker.input_queue.put_nowait(None)
            except queue.Full: pass
        print("[MAIN] Shutdown complete.", file=sys.stderr)