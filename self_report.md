# Self-report on Nada project

## Architectural choices:

1.  ### Speculative execution

    By having the minimisation of local latency as our primary goal, a key latency window to cut was the "dead-time" between the end of user speech until the confirmation of the end of the user's turn. Typically, this would be about 850ms of dead-time, where no processes are happening. Naively, this adds 850ms of additional latency before we even begin any processes, massively impacting our time-to-first-audio.

    `Advantages`: By "assuming" that the user is finished at an earlier time threshold, we can start processes and gain a headstart, saving hundreds of ms in our pipeline.

    `Disadvantages`: We currently suffer from the system discarding the rest of the user's speech after the initial silence threshold has been reached. This is something we can look into to fix, perhaps by saving/storing the speech and appending it once the user's speech resumes before the 850ms mark. Some sort of cancellation strategy needs to be conceived.

2.  ### No usage of frameworks like pipecat or vogent:

    Upon research, we found that there exist frameworks like pipecat and vogent which provide many beneficial features and functions for voice pipelines, removing the need to make things from scratch.

    `Advantages`: Potentially reducing overhead when making the system, introducing cleaner code.

    `Disadvanatges`: The frameworks mainly supported paid TTS services besides piper-tts. Though I believe piper-tts would actually produce better streaming latency performance than kokoro on a local system, the piper-tts voice was audibly worse than kokoro, and hence would defeat some aspect of human experience. Furthermore, we are able to swap out components manually, without having to fight for framework compatability. We can also demonstrate more control on making small tweaks to drive performance, and we are not restricted by the framework's intended architecture thus giving us room to experiment.

3.  ### No usage of NLP libraries

    While developing the system, I experimented with libraries like NLTK and spacy. These would be used by the TTS worker to tokenise the incoming text properly to avoid abrupt breaks in sentences.

    `Advantages`: By not using spacy and nltk, but rather using a custom rule-based logic, we save on latency grounds, as the libraries produced too much latency cost for them to be worth the grammatical safety.

    `Disadvantages`: Though our system can be mostly accurate, it is not fail-safe, and can still often break up at abrupt places in sentences. Though it is masked by the high-performance of our rapid playback speed to play subsequeunt audio chunks almost instantly, we can try to find more secure tokenisation libraries which offer ultra-fast computation.

4.  ### Instant interruption vs self-interruption:

    This is a difficult issue that I couldn't effectively tackle. Since from the start, our architecture aimed at ultra-low latency speech and instant interruption, the way our system works is that while the agent is speaking, if there is any input sound which goes above the vad confidence threshold, it will signal the interruption event and the agent will stop speaking.

    `Advantages`: Through this logic, our system provides instant, "barge-in" interruption. We don't have to wait for the agent to finish their sentence or response, but rather we can immediately start speaking and the agent will be fully aware of our speech, listening and preparing for a response. It serves as a real-time, palpable human experience, but only on headphones.

    `Disadvantages`: Unfortunately, the system is unable to distinguish it's own sound from the user's. I tried several approaches. Using flags to prevent self-interruption would inadvertently prevent interruption entirely, as the agent would wait to finish speaking. I tried to use a "signature-matching" capability, however this proved a bit difficult with unreliable performance, as voices are not unique as would something like a fingerprint be. Hence, even if it was to work properly, in the case that the agent voice sounds similar to the human's voice, our logic would inherently fail. A possible solution to this is through a push-to-talk mechanism. This would instantly fix self-interruption while also retaining the regular human interruption capability, as we would not signal an interrupt based on voice detection, but rather the press of a button. However, this may undermine the user experience. I'd need further exploration into this issue. We need to look into AEC further in-depth than I did.

5.  ### No API or web usage:

    This is something I saw in many state-of-the-art models during my research. Whether it was telnyx, vocalis or many of the other high-performance, production-grade s2s systems I saw, many of them were using APIs/websockets and conducting the whole pipeline as a client-server interaction over the network, rather than a fully local system. This means that they'd need a central server with intense, capable hardware like an A100, and run all the computation centrally and deal with user requests, while the client side would have a simple frontend and be primarily UI focused.

    `Advantages`: The advantage of a local system is that we avoid any privacy concerns or restrictions faced when sending data, especially sensitive company data, over a network. We reduce the chance of any vulnerabilities appearing in our system, and also avoid having to invest ourselves into high cost data centers and expensive GPUs and hardware, as well as recurring costs from having to maintain this setup. We also protect ourselves from any network outages or network latency harming our performance.

    `Disadvantages`: We don't get access to absolute state-of-the-art models and systems from OpenAI, Google etc. We also may have to address scalability concerns, as companies would have to host the powerful hardware themselves and every company may not be willing to do this and would rather pay for a cloud-based service.

    I personally think we could be okay with this, as long as we make sure that in further production stages, we make sure that our system is absolutely reliable, and provide them with a great customer experience from our side and address any concerns they face. We also have the ability to refine our solutions, specifically catering to client needs - there may be a feature that client A prioritises over another feature that client B prioritises, and hence by having a local system, we'd be flexible in our approach and would be able to address client requirements individually. Basically, instead of a one-size-fit-all solution, we can modify our system for companies to prioritise their wants, which I don't see possible on a cloud based service.

6.  ### Used silero vad rather than webrtcvad

    This is a simple one - silero vad returns a probability value, webrtcvad returns a boolean.

    `Advantages`: We are able to tune the confidence thresholds for speech detection

    `Disadvantages`: Not any evident ones.

7.  ### Producer - Consumer audio chunk management

    Our `main_tts_thread` produces audio chunks and places them onto a queue, and `_audio_callback` method consumes these chunks for playback.

    `Advantages`: We are able to decouple audio production from audio consumption, and hence we do not fall victim to blocking processes, and helps us ensure that the playback thread always has audio chunks to play and doesn't stutter through waiting.

    `Disadvantages`: Not any evident ones.

8.  ### 8 bit model quantisation:

    We have used 8bit quantisation through the `BitsAndBytesConfig` library.

    `Advantages`: Allows us to prioritise latency and drive performance. Worked well for a local prototype

    `Disadvantages`: Can lead to decreased accuracy in the model's performance. We can possibly remove 8bit quantisation if we feel the hardware is capable enough of running the models as such without taking a hit for latency.

9.  ### Multithreaded approach

    Using different threads for independent areas of the system

    `Advantages`: Fairly evident - we can concurrently run processes which don't need to occur sequentially. Better than single-threaded `asyncio` and less overhead than `multiprocessing`.

    `Disadvantages`: No evident flaws.

## Personal journey

The whole process of making this from scratch was indeed quite fun. Though I had very little knowledge on how to orchestrate and manually code up the complex architecture with lots of multithreading, streams and dealing with blocking/non-blocking processes and assigning different priorities to threads, I constructed my workflow to have Gemini as the main coder, while I would do in-depth research about existing solutions and their pros/cons, do high-level, architectural thinking, trace the stack and debug errors/unexpected performance, and try to conceive novel, untapped ideas in pursuit of breakthroughs in system performance. In a sense, I was the architect, and LLMs were the engineers.

For further progress on this project, I'd love to first see how it performs on more powerful hardware. I'd also want to tackle the self-interruption vs instant interruption issue, and take your guidance on where our logic fails. Once we fix this, we can explore ways to offer improved system customisation, and look at fine-tuning / RAG methods for individual clients, and package everything nicely for them. Also, as the voice AI landscape is changing quite fast, we should be aware of upcoming, better, open-source models which can provide us with better performance and features. We also need to look at function calling, and see how we can implement this to give our agent access to company tools.

All-in-all, I absolutely loved this project, and would be really happy to continue working and developing this further with you guys.
Many thanks for this amazing opportunity!
