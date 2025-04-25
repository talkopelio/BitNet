FROM ubuntu:22.04 AS build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    curl \
    wget \
    lsb-release \
    cmake \ 
    software-properties-common \
    gnupg \
    && rm -rf /var/lib/apt/lists/*


RUN git clone --recursive https://github.com/talkopelio/BitNet.git; 
WORKDIR /BitNet
RUN git pull

# Install Python dependencies
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN pip3 install -r requirements.txt

# Install FastAPI and Uvicorn for HTTP server


# Build the project
RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
RUN python3 setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

RUN pip3 install fastapi uvicorn
# Create API server file
COPY server.py /BitNet/server.py

# Expose the API port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

