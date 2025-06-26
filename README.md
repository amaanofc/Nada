Nada: A State-of-the-Art, Low-Latency Conversational AI
This repository contains the full source code for Nada, a highly responsive, real-time, voice-to-voice conversational AI agent. Built using Python, this project demonstrates a sophisticated, multi-threaded architecture designed to minimize latency and provide a natural, fluid conversational experience.

✨ Key Features
This agent is more than just a simple script; it's an engine built with several state-of-the-art techniques:

🎙️ Voice Activity Detection (VAD): Intelligently detects when the user starts and stops speaking, enabling natural turn-taking.

🧠 Speculative Execution: The agent "thinks ahead" by starting to process the user's speech during natural pauses, dramatically reducing perceived latency.

⚡ Hyper-Low Latency Pipeline: A fully parallelized, multi-threaded architecture ensures that language model processing, text-to-speech synthesis, and audio playback happen concurrently to minimize bottlenecks.

🌊 Seamless Streaming Audio: Utilizes a non-blocking, callback-based audio stream combined with a "Prime & Parallelize" generation strategy. This eliminates the usual gaps between spoken sentences, creating a single, uninterrupted flow of speech.

🏃‍♀️ Dynamic Speech Rate: The agent intelligently adapts its speaking speed based on the conversational rhythm, speaking faster in quick exchanges and more deliberately during thoughtful pauses.

🗣️ Instant Interruption: The user can barge in and interrupt the agent at any time, and the agent will instantly stop speaking and listen, just like a natural human conversation partner.

🔇 Audio Quality Gate: Automatically filters out and ignores low-volume background noise or accidental non-speech sounds to prevent erroneous activations.

🏛️ System Architecture
The agent's performance is achieved through a carefully orchestrated, multi-threaded architecture designed to keep the pipeline full and eliminate blocking operations wherever possible.


📋 Prerequisites
Hardware: An NVIDIA GPU with CUDA support (e.g., RTX 30-series or newer) is highly recommended for acceptable performance.

Software:

Python 3.10+

An NVIDIA driver with a compatible CUDA Toolkit version installed. You can check your CUDA version by running nvidia-smi in your terminal.

An internet connection for downloading models on the first run.

🚀 Installation & Setup (Definitive Guide)
This guide ensures you install the GPU-accelerated version of PyTorch for maximum performance.

Step 1: Clone the Repository
git clone <your-repo-url>
cd nada-agent

Step 2: Create and Activate a Virtual Environment
This keeps your project dependencies isolated.

On Windows:

python -m venv .venv
.\.venv\Scripts\activate

On macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

Step 3: Install PyTorch with CUDA (CRITICAL STEP)
You must install the version of PyTorch that matches your system's CUDA installation for GPU acceleration.

Go to the official PyTorch website: https://pytorch.org/get-started/locally/

Use the "Get Started" configuration tool to select the correct options for your system (e.g., Stable, Windows/Linux, Pip, Python, your CUDA version).

Copy the generated command and run it in your activated terminal. The following should most probably work:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Step 4: Install All Other Dependencies
Once PyTorch is installed correctly, install the rest of the required packages from the requirements.txt file.

pip install -r requirements.txt

You are now fully set up!

▶️ How to Run
With your virtual environment activated, simply run the main agent script:

python agent.py

The agent will initialize all the models. The first time you run it, this may take a minute as models are downloaded to your cache. Once you see the >>> AGENT IS LISTENING. SPEAK NOW. <<< message, you can begin your conversation!

Press Ctrl+C in the terminal to stop the agent gracefully.

⚙️ Configuration
You can easily tweak the agent's performance and behavior by modifying the variables in the AgentConfig class at the top of agent.py. This includes changing the VAD sensitivity, speech speed, and audio quality gate thresholds.
