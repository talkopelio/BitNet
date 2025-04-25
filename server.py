import os
import json
import subprocess
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
import asyncio
import tempfile
import time
import logging
from contextlib import asynccontextmanager
import select

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bitnet-server")

# Model worker class to manage persistent llama-cli process
class LlamaProcess:
    def __init__(self, model_path, threads, context_size=2048):
        self.model_path = model_path
        self.threads = threads
        self.context_size = context_size
        self.process = None
        self.init_process()
    
    def init_process(self):
        logger.info("Initializing llama-cli process")
        if self.process is not None:
            logger.info("Stopping existing process before reinitializing")
            self.stop_process()
        
        # Start the process in interactive mode, without conversation flag
        command = [
            'python3',
            'run_inference.py',
            '-m', self.model_path,
            '-t', str(self.threads),
            '-c', str(self.context_size),
        ]
        logger.info(f"Starting with command: {' '.join(command)}")
        
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffering
                universal_newlines=True  # Ensures text mode
            )
            logger.info(f"Successfully started process with PID {self.process.pid}")
            # Wait for the initial prompt indicator "> "
            self._wait_for_prompt(timeout=10)
        except Exception as e:
            logger.error(f"Failed to start process: {str(e)}")
            stderr_output = ""
            if self.process and self.process.stderr:
                try:
                    stderr_output = self.process.stderr.read()
                except Exception as read_err:
                    stderr_output = f"Error reading stderr: {read_err}"
            logger.error(f"stderr content: {stderr_output}")
            self.process = None  # Ensure process is None if startup fails
            raise
    
    def _wait_for_prompt(self, timeout=10):
        """Wait for the llama-cli prompt indicator '>'."""
        logger.debug("Waiting for initial prompt indicator '>'")
        output_buffer = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read()
                logger.error(f"Process terminated unexpectedly during init. Exit code: {self.process.poll()}. Stderr: {stderr_output}")
                raise RuntimeError("Llama process terminated unexpectedly during initialization.")

            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)  # Non-blocking read check
            if ready:
                char = self.process.stdout.read(1)
                if not char:  # EOF
                    logger.warning("EOF reached on stdout while waiting for prompt.")
                    time.sleep(0.1)  # Avoid busy-waiting if EOF is temporary?
                    continue
                output_buffer += char
                # Check if the buffer ends with the prompt indicator "> "
                if output_buffer.endswith("> "):
                    logger.info(f"Initial prompt received. Output skipped: {output_buffer[:100]}...")
                    return True
            else:
                # No output ready, continue loop or timeout
                pass  # Wait briefly before next check? select handles timeout

        logger.warning("Timed out waiting for initial prompt indicator '>'")
        stderr_output = self.process.stderr.read()  # Read stderr on timeout
        logger.error(f"Stderr after timeout: {stderr_output}")
        raise TimeoutError("Timed out waiting for llama-cli prompt indicator.")
    
    async def generate(self, prompt, max_tokens=256):
        request_id = f"req-{int(time.time() * 1000)}"
        logger.info(f"{request_id}: Generate request received")
        
        if self.process is None or self.process.poll() is not None:
            logger.warning(f"{request_id}: Process not running or terminated, reinitializing")
            try:
                self.init_process()
            except Exception as e:
                logger.error(f"{request_id}: Failed to reinitialize process: {e}")
                raise HTTPException(status_code=503, detail="Model process failed to initialize.")
        
        try:
            # Send prompt
            logger.info(f"{request_id}: Sending prompt (first 100 chars): {prompt[:100]}...")
            self.process.stdin.write(prompt + "\n")
            self.process.stdin.flush()
            
            # Read response until the next prompt indicator "> "
            logger.info(f"{request_id}: Waiting for response")
            response_buffer = ""
            start_time = time.time()
            timeout = 90  # seconds
            
            while time.time() - start_time < timeout:
                if self.process.poll() is not None:
                    stderr_output = self.process.stderr.read()
                    logger.error(f"{request_id}: Process terminated during generation. Exit code: {self.process.poll()}. Stderr: {stderr_output}")
                    raise HTTPException(status_code=500, detail="Model process terminated unexpectedly.")
                
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if ready:
                    char = self.process.stdout.read(1)
                    if not char:
                        logger.warning(f"{request_id}: EOF reached on stdout while reading response.")
                        await asyncio.sleep(0.1)  # Wait a bit before checking again
                        continue
                    response_buffer += char
                    if response_buffer.endswith("> "):
                        logger.info(f"{request_id}: Response complete indicator found.")
                        break  # Got the full response terminated by the prompt
                else:
                    # No data available, wait briefly
                    await asyncio.sleep(0.01)
            
            if not response_buffer.endswith("> "):
                logger.error(f"{request_id}: Timeout waiting for model response completion (> indicator).")
                # Attempt to kill/restart
                self.stop_process()
                raise HTTPException(status_code=500, detail="Model response timeout (did not receive completion indicator).")
            
            elapsed = time.time() - start_time
            logger.info(f"{request_id}: Got response in {elapsed:.2f} seconds")
            
            # Clean the response: remove the prompt indicator and the original prompt echo if present
            response = response_buffer.strip()
            if response.endswith(">"): response = response[:-1].strip()  # Remove trailing ">"
            
            # The interactive mode often echoes the input prompt first.
            # We need to reliably remove it. Assuming it appears exactly once at the start.
            # It might be safer to look for the assistant start token.
            # Simplified cleaning for now:
            # Find the *last* occurrence of the input prompt in the buffer,
            # as the model might generate something similar.
            # This is still fragile. A better approach uses the actual output structure.
            # Let's assume the output starts immediately after the echoed prompt + newline
            prompt_echo = prompt + "\n"
            if response.startswith(prompt_echo):
                response = response[len(prompt_echo):]
            
            # Remove potential trailing "> " artifacts again after stripping prompt
            response = response.replace("> ", "").strip()
            
            truncated_response = response[:100] + "..." if len(response) > 100 else response
            logger.info(f"{request_id}: Cleaned Response (first 100 chars): {truncated_response}")
            
            return response
        
        except BrokenPipeError:
            logger.error(f"{request_id}: Broken pipe error communicating with llama-cli. Restarting process.")
            self.stop_process()  # Clean up the dead process
            raise HTTPException(status_code=500, detail="Model process connection error (broken pipe).")
        except Exception as e:
            logger.error(f"{request_id}: Error during generation: {str(e)}", exc_info=True)
            # Check process state
            if self.process and self.process.poll() is not None:
                logger.error(f"{request_id}: Process exited with code {self.process.poll()}. Trying to read stderr.")
                try:
                    stderr_output = self.process.stderr.read()
                    logger.error(f"{request_id}: Process stderr: {stderr_output}")
                except Exception as read_err:
                    logger.error(f"{request_id}: Could not read stderr: {read_err}")
                self.process = None  # Mark as dead
            raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
    
    def stop_process(self):
        logger.info("Stopping llama-cli process")
        if self.process is not None and self.process.poll() is None:
            try:
                logger.info(f"Sending terminate signal to PID {self.process.pid}")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                    logger.info("Process terminated successfully.")
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate, sending kill signal.")
                    self.process.kill()
                    self.process.wait(timeout=2)  # Wait briefly for kill
                    logger.info("Process killed.")
            except Exception as e:
                logger.error(f"Error stopping process: {str(e)}")
            finally:
                # Ensure stdin/stdout/stderr are closed? Popen should handle this.
                self.process = None
        else:
            logger.info("Process already stopped or not running.")

# Global settings
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."
EXECUTABLE_PATH = "build/bin/llama-cli"
DEFAULT_THREADS = int(os.environ.get("MAX_THREADS", 4))

logger.info(f"Starting server with {DEFAULT_THREADS} threads")

# Verify executable exists
if not os.path.exists(EXECUTABLE_PATH):
    logger.error(f"Could not find executable at {EXECUTABLE_PATH}")
    raise FileNotFoundError(f"Could not find executable at {EXECUTABLE_PATH}.")
else:
    logger.info(f"Found executable at {EXECUTABLE_PATH}")

# Single global model process instance
model_process: Optional[LlamaProcess] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_process
    # Initialize the single model process on startup
    logger.info("Initializing model process...")
    try:
        model_process = LlamaProcess(MODEL_PATH, DEFAULT_THREADS)
        logger.info("Model process initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize model process on startup: {e}", exc_info=True)
        # Optionally exit or prevent FastAPI from starting fully
        raise RuntimeError("Failed to initialize the core model process.") from e
    
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down model process...")
    if model_process:
        model_process.stop_process()
    logger.info("Model process stopped.")

app = FastAPI(title="BitNet LLM API Server", lifespan=lifespan)

class GenerationRequest(BaseModel):
    prompt: str
    # system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT # Removed system_prompt from request
    max_tokens: Optional[int] = 256
    # Removed temperature, top_p, context_size, threads, conversation_mode as they are fixed in the process start
    # Re-add if dynamic control is needed via command line args manipulation (complex)

class GenerationResponse(BaseModel):
    generated_text: str

@app.get("/")
async def root():
    # logger.info("Root endpoint accessed")  # Reduced logging
    return {"message": "BitNet LLM API Server is running"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    request_id = f"req-{int(time.time() * 1000)}"
    logger.info(f"{request_id}: Generate text request received for prompt: {request.prompt[:50]}...")
    
    if model_process is None:
        logger.error(f"{request_id}: Model process is not available.")
        raise HTTPException(status_code=503, detail="Model process not initialized or failed.")
    
    try:
        # Prepare simplified prompt using the default system prompt
        # Ensure there's a clear separation for the model. Newlines are important.
        full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\\nUser: {request.prompt}\\nAssistant:"

        logger.info(f"{request_id}: Calling generate on model process")
        start_time = time.time()
        
        # Use the single global model process instance
        response_text = await model_process.generate(full_prompt, request.max_tokens)
        
        elapsed = time.time() - start_time
        logger.info(f"{request_id}: Generate completed in {elapsed:.2f} seconds")
        
        # Response text cleaning is now handled within LlamaProcess.generate
        
        # Prepare response
        result = {"generated_text": response_text}
        logger.info(f"{request_id}: Response ready.")
        
        return result
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly (e.g., from generate method)
        raise http_exc
    except Exception as e:
        logger.error(f"{request_id}: Unexpected error in /generate endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on 0.0.0.0:8000")
    # Running with reload=False is typical for production/simpler setups
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False) 