# DeepSeek-R1 Serverless Deployment with SGLang

This project provides a serverless deployment of the DeepSeek-R1-Distill-Qwen-32B model using SGLang on RunPod. It leverages tensor parallelism for efficient model serving.

## Features

- DeepSeek-R1-Distill-Qwen-32B model integration
- SGLang optimization for faster inference
- Tensor Parallelism support (2 GPUs by default)
- Detailed logging system
- RunPod serverless deployment
- Docker containerization

## Requirements

- RunPod account with GPU access
- Docker for building the image
- At least 2 GPUs with 24GB+ VRAM each
- CUDA 12.1 compatible environment

## Project Structure

```
.
├── Dockerfile           # Docker configuration
├── handler.py          # Main RunPod handler and SGLang server
└── README.md           # This documentation
```

## Quick Start

### 1. Docker Image Options

#### Option A: Use Pre-built Image (Recommended for Quick Start)
You can directly use our pre-built image:
```
carlea/deepseek-r1:latest
```
This image is ready to use and includes all necessary dependencies.

#### Option B: Build Your Own Image
If you need to customize the image, you can build it yourself:

```bash
# Build the image
docker build --platform linux/amd64 -t your-username/deepseek-r1:latest .

# Push to Docker Hub
docker push your-username/deepseek-r1:latest
```

### 2. RunPod Deployment

1. Go to RunPod.io
2. Create a new Serverless Endpoint
3. Select a template with at least 2 GPUs
4. Use either:
   - Pre-built image: `carlea/deepseek-r1:latest`
   - Or your custom image: `your-username/deepseek-r1:latest`
5. Set the following environment variables:
   - `NUM_GPUS=2` (or the number of GPUs you want to use)

### 3. Making Requests

#### API Endpoint

The endpoint URL will be provided by RunPod in the format:
```
https://api.runpod.ai/v2/{endpoint-id}/run
```

#### Authentication

Add your RunPod API key in the request headers:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

#### Request Format

The API accepts POST requests with the following JSON structure:

```json
{
    "input": {
        "prompt": "Your text prompt here",
        "max_length": 1024,        // Optional, default: 1024
        "temperature": 0.7,        // Optional, default: 0.7
    }
}
```

##### Parameters Explanation:
- `prompt` (required): The input text to generate from
  - Type: string
  - Example: "Write a story about a space adventure"

- `max_length` (optional):
  - Type: integer
  - Default: 1024
  - Range: 1-4096
  - Description: Maximum number of tokens to generate

- `temperature` (optional):
  - Type: float
  - Default: 0.7
  - Range: 0.0-1.0
  - Description: Controls randomness in generation (0.0 = deterministic, 1.0 = creative)

#### Example Requests

1. **Basic Request (Python)**:
```python
import requests

url = "https://api.runpod.ai/v2/{endpoint-id}/run"
headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "input": {
        "prompt": "Write a story about a space adventure"
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

2. **Advanced Request (Python)**:
```python
data = {
    "input": {
        "prompt": "Write a technical documentation about quantum computing",
        "max_length": 2048,
        "temperature": 0.3  # Lower temperature for more focused technical content
    }
}
```

3. **cURL Example**:
```bash
curl -X POST \
  https://api.runpod.ai/v2/{endpoint-id}/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
        "prompt": "Write a story about a space adventure",
        "max_length": 1024,
        "temperature": 0.7
    }
  }'
```

#### Response Format

The API returns JSON responses in the following format:

```json
{
    "id": "request-id",
    "status": "completed",
    "output": {
        "generated_text": "Generated text response here..."
    }
}
```

##### Error Response:
```json
{
    "id": "request-id",
    "status": "failed",
    "error": {
        "message": "Error description here"
    }
}
```

#### Rate Limiting and Quotas

- Requests are processed based on your RunPod plan
- Default timeout: 30 seconds
- Maximum concurrent requests: Based on GPU availability

## Technical Details

### SGLang Server

The project uses SGLang for optimized inference:
- Automatic tensor parallelism across GPUs
- Optimized attention mechanisms
- Efficient memory management
- Support for both NVIDIA and AMD GPUs

### Logging System

The application includes comprehensive logging:
- Server startup and configuration
- Model loading progress
- Request handling
- Error tracking
- Server health monitoring

### Error Handling

The system includes robust error handling for:
- Server startup failures
- Model loading issues
- Request processing errors
- Server health checks
- Graceful shutdowns

## Troubleshooting

### Common Issues

1. **Worker Unhealthy Status**
   - Check the logs for startup errors
   - Verify GPU availability
   - Ensure sufficient VRAM

2. **Model Loading Failures**
   - Check internet connectivity
   - Verify Hugging Face access
   - Ensure sufficient disk space

3. **Request Timeouts**
   - Adjust the max_length parameter
   - Check server load
   - Monitor GPU memory usage

### Accessing Logs

To view detailed logs:
1. Go to RunPod dashboard
2. Select your endpoint
3. Click on "Logs" for the specific worker
4. Look for entries with timestamps and log levels

## Performance Optimization

For optimal performance:
1. Adjust `NUM_GPUS` based on your hardware
2. Monitor GPU memory usage
3. Adjust batch sizes if needed
4. Consider request queue settings in RunPod

## Security Considerations

- The model runs in an isolated container
- API endpoints require RunPod authentication
- Model weights are downloaded securely from Hugging Face
- Environment variables are used for sensitive configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. The DeepSeek-R1 model has its own license - please check Hugging Face for details.

## Acknowledgments

- DeepSeek AI for the model
- SGLang team for the optimization framework
- RunPod for the serverless infrastructure

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review RunPod documentation
3. Open an issue on GitHub
4. Contact RunPod support for platform-specific issues 