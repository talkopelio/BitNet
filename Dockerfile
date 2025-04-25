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


RUN git clone --recursive https://github.com/talkopelio/BitNet.git
WORKDIR /BitNet


# Copy the source code

# Install python dependencies
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN pip3 install -r requirements.txt

# Build the project
RUN huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
RUN python3 setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s





# # Download a default model (you can override this with volume mounts)
# RUN mkdir -p models/BitNet-b1.58-2B-4T && \
#     python3 -m huggingface_hub download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

# # Create a lighter runtime image
# FROM ubuntu:22.04 AS runtime

# # Install runtime dependencies
# RUN apt-get update && \
#     apt-get install -y \
#     python3 \
#     python3-pip \
#     libcurl4-openssl-dev \
#     curl \
#     wget \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Install Flask and required packages for API
# RUN pip3 install flask gunicorn requests pydantic

# # Copy the built binaries and necessary files from the build stage
# COPY --from=build /app/build /app/build
# COPY --from=build /app/bin /app/bin
# COPY --from=build /app/models /app/models
# COPY --from=build /app/run_inference.py /app/run_inference.py
# COPY --from=build /app/requirements.txt /app/requirements.txt

# # Create API server
# COPY api_server.py /app/api_server.py

# # Set environment variables
# ENV LC_ALL=C.utf8
# ENV HOST=0.0.0.0
# ENV PORT=8080
# ENV MODEL_PATH=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# # Expose the port
# EXPOSE 8080

# # Healthcheck
# HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

# # Run the API server
# CMD ["python3", "api_server.py"] 