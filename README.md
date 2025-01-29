# DeepSeek-R1 Serverless Deployment with SGLang

🔒 **Own Your AI, Protect Your Data**

This project allows you to deploy your own private AI API server, giving you complete control over your data and privacy. Think of it as having your own ChatGPT-like service, but where you own and control everything:

- ✅ **Full Privacy**: Your data never leaves your infrastructure
- ✅ **Complete Control**: You own and manage the API
- ✅ **Enterprise-Ready**: Perfect for businesses requiring data sovereignty
- ✅ **High Performance**: Comparable to OpenAI's GPT models
- ✅ **Cost-Effective**: Pay only for what you use with RunPod's serverless infrastructure

No more worrying about:
- ❌ Data being used to train other models
- ❌ Privacy policies changing
- ❌ Vendor lock-in
- ❌ API availability issues

## What Makes This Special?

1. **Privacy-First**: Your data stays within your control, perfect for sensitive business information
2. **Enterprise Security**: Deploy in your own secure environment
3. **Customizable**: Adjust the model parameters to your needs
4. **Cost-Effective**: Serverless architecture means you only pay for actual usage
5. **High Performance**: Optimized with SGLang for fast inference

## Cost Comparison

| Service | Cost per 1K tokens | Privacy | Control | Lock-in |
|---------|-------------------|----------|----------|----------|
| This Solution | $0.005-0.01* | Full | Full | None |
| OpenAI GPT-4 | $0.03 | Limited | Limited | Yes |
| Claude 2 | $0.03 | Limited | Limited | Yes |

\* Actual cost depends on RunPod GPU pricing and usage patterns

### Why Choose This Over Other Services?

1. **Privacy & Security**
   - Your data never leaves your infrastructure
   - Perfect for sensitive business data
   - Compliant with data protection regulations

2. **Cost Benefits**
   - No markup on API calls
   - Pay only for compute resources
   - Scale down to zero when not in use

3. **Performance**
   - Similar capabilities to leading models
   - Optimized for low latency
   - Customizable for your needs

4. **Independence**
   - No vendor lock-in
   - Full control over the API
   - Deploy anywhere you want

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

## Quick Start Guide

### Option 1: Ready-to-Use Deployment (Recommended)

1. **Create a RunPod Account**
   - Go to [RunPod.io](https://runpod.io)
   - Sign up for an account
   - Add your payment method

2. **Deploy the API**
   - Go to Serverless > Create Endpoint
   - Select a template with 2+ GPUs
   - Use our pre-built image: `carlea/deepseek-r1:latest`
   - Set `NUM_GPUS=2` in environment variables

3. **Start Using Your API**
```python
import requests

   # Your RunPod API endpoint
   url = "https://api.runpod.ai/v2/{your-endpoint-id}/run"
   
   # Your API key
headers = {
       "Authorization": "Bearer YOUR_RUNPOD_API_KEY"
}

   # Sample request
   response = requests.post(url, 
       json={
    "input": {
               "prompt": "What is artificial intelligence?"
    }
       },
       headers=headers
   )

print(response.json())
```

## API Reference

### Endpoint Format
```
https://api.runpod.ai/v2/{endpoint-id}/run
```

### Authentication
Add your RunPod API key in the request headers:
```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

### Request Format
```json
{
    "input": {
        "prompt": "Your text prompt here",
        "max_length": 1024,        // Optional, default: 1024
        "temperature": 0.7         // Optional, default: 0.7
    }
}
```

### Parameters
- `prompt` (required)
  - Your input text
  - Type: string
  - Example: "Write a story about AI"

- `max_length` (optional)
  - Maximum number of tokens to generate
  - Type: integer
  - Default: 1024
  - Range: 1-4096

- `temperature` (optional)
  - Controls creativity vs consistency
  - Type: float
  - Default: 0.7
  - Range: 0.0-1.0
  - 0.0 = deterministic, 1.0 = creative

### Example Requests

1. **Basic Generation**
```python
data = {
    "input": {
        "prompt": "Write a story about AI"
    }
}
```

2. **Technical Content**
```python
data = {
    "input": {
        "prompt": "Explain quantum computing",
        "temperature": 0.3,  # Lower for technical content
        "max_length": 2048   # Longer response
    }
}
```

3. **Creative Writing**
```python
data = {
    "input": {
        "prompt": "Write a sci-fi story",
        "temperature": 0.9,  # Higher for creativity
        "max_length": 1500
    }
}
```

### Response Format
```json
{
    "id": "request-id",
    "status": "completed",
    "output": {
        "generated_text": "Generated response here..."
    }
}
```

### Error Response
```json
{
    "id": "request-id",
    "status": "failed",
    "error": {
        "message": "Error description"
    }
}
```

## Use Cases

### 1. Enterprise & Business
- Private customer support AI
- Internal documentation assistant
- Secure data analysis
- Compliance-friendly AI integration

### 2. Healthcare
- Patient data processing
- Medical research assistance
- Healthcare documentation
- Clinical decision support

### 3. Financial Services
- Secure transaction analysis
- Risk assessment
- Customer service automation
- Compliance documentation

### 4. Legal
- Document analysis
- Case research
- Contract review
- Legal research assistance

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