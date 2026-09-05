# Bedrock Image Analyzer
# Main script for sending images to AWS Bedrock models for transcription and analysis

import base64
import os
from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
from datetime import datetime, timezone
import uuid
#import router


def get_image_size_mb(image: Image.Image, format: str) -> float:
    """
    Calculate the size of an image in megabytes.
    
    Args:
        image: PIL Image object
        format: Image format (e.g., 'JPEG', 'PNG')
    
    Returns:
        Size in megabytes as a float
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    size_bytes = buffer.tell()
    return size_bytes / (1024 * 1024)


def reduce_image_size(image: Image.Image, target_size_mb: float = 3.5) -> tuple:
    """
    Reduce image size using progressive optimization techniques.
    
    Args:
        image: PIL Image object to optimize
        target_size_mb: Target size in megabytes (default: 3.5)
    
    Returns:
        Tuple of (optimized_image, list_of_applied_techniques)
    
    Raises:
        ValueError: If unable to reduce image below target size
    """
    optimized_image = image.copy()
    applied_techniques = []
    original_format = image.format if image.format else 'JPEG'
    current_format = original_format
    
    # Check current image size
    current_size = get_image_size_mb(optimized_image, current_format)
    
    if current_size <= target_size_mb:
        return optimized_image, applied_techniques
    
    # Strategy 1: Apply quality reduction for JPEG (95→85→75→65)
    if current_format in ['JPEG', 'JPG'] or original_format in ['JPEG', 'JPG']:
        for quality in [95, 85, 75, 65]:
            buffer = BytesIO()
            optimized_image.save(buffer, format='JPEG', quality=quality)
            size_mb = buffer.tell() / (1024 * 1024)
            
            if size_mb <= target_size_mb:
                buffer.seek(0)
                optimized_image = Image.open(buffer)
                optimized_image.load()
                current_format = 'JPEG'
                applied_techniques.append(f"Quality reduction to {quality}")
                print(f"Applied quality reduction to {quality}, size: {size_mb:.2f}MB")
                return optimized_image, applied_techniques
    
    # Strategy 2: Apply resolution scaling (20% increments)
    width, height = optimized_image.size
    min_dimension = 800  # Don't scale below this on longest dimension
    
    for scale_factor in [0.8, 0.6, 0.4, 0.2]:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Check if we're maintaining minimum dimension
        if max(new_width, new_height) < min_dimension:
            continue
        
        scaled_image = optimized_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Try with current format first
        size_mb = get_image_size_mb(scaled_image, current_format)
        
        if size_mb <= target_size_mb:
            optimized_image = scaled_image
            applied_techniques.append(f"Resolution scaling to {int(scale_factor * 100)}%")
            print(f"Applied resolution scaling to {int(scale_factor * 100)}%, size: {size_mb:.2f}MB")
            return optimized_image, applied_techniques
    
    # Strategy 3: Apply format optimization (PNG→JPEG)
    if current_format == 'PNG':
        # Convert to RGB if necessary (PNG might have alpha channel)
        if optimized_image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', optimized_image.size, (255, 255, 255))
            if optimized_image.mode == 'P':
                optimized_image = optimized_image.convert('RGBA')
            rgb_image.paste(optimized_image, mask=optimized_image.split()[-1] if optimized_image.mode in ('RGBA', 'LA') else None)
            optimized_image = rgb_image
        elif optimized_image.mode != 'RGB':
            optimized_image = optimized_image.convert('RGB')
        
        current_format = 'JPEG'
        size_mb = get_image_size_mb(optimized_image, current_format)
        
        if size_mb <= target_size_mb:
            applied_techniques.append("Format optimization (PNG to JPEG)")
            print(f"Applied format optimization (PNG to JPEG), size: {size_mb:.2f}MB")
            return optimized_image, applied_techniques
        
        # Try with quality reduction after format conversion
        for quality in [85, 75, 65]:
            buffer = BytesIO()
            optimized_image.save(buffer, format='JPEG', quality=quality)
            size_mb = buffer.tell() / (1024 * 1024)
            
            if size_mb <= target_size_mb:
                buffer.seek(0)
                optimized_image = Image.open(buffer)
                optimized_image.load()
                applied_techniques.append("Format optimization (PNG to JPEG)")
                applied_techniques.append(f"Quality reduction to {quality}")
                print(f"Applied format optimization and quality reduction to {quality}, size: {size_mb:.2f}MB")
                return optimized_image, applied_techniques
    
    # Strategy 4: Apply grayscale conversion as last resort
    if optimized_image.mode != 'L':
        grayscale_image = optimized_image.convert('L')
        size_mb = get_image_size_mb(grayscale_image, current_format)
        
        if size_mb <= target_size_mb:
            optimized_image = grayscale_image
            applied_techniques.append("Grayscale conversion")
            print(f"Applied grayscale conversion, size: {size_mb:.2f}MB")
            return optimized_image, applied_techniques
        
        # Try grayscale with resolution scaling
        width, height = grayscale_image.size
        for scale_factor in [0.8, 0.6, 0.4]:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            if max(new_width, new_height) < min_dimension:
                continue
            
            scaled_gray = grayscale_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            size_mb = get_image_size_mb(scaled_gray, current_format)
            
            if size_mb <= target_size_mb:
                optimized_image = scaled_gray
                applied_techniques.append("Grayscale conversion")
                applied_techniques.append(f"Resolution scaling to {int(scale_factor * 100)}%")
                print(f"Applied grayscale conversion and scaling to {int(scale_factor * 100)}%, size: {size_mb:.2f}MB")
                return optimized_image, applied_techniques
    
    # If we've exhausted all strategies, raise an error
    final_size = get_image_size_mb(optimized_image, current_format)
    raise ValueError(
        f"Unable to reduce image below {target_size_mb}MB after applying all optimization techniques. "
        f"Final size: {final_size:.2f}MB. Try manually resizing the image."
    )


def load_and_encode_image(image_path: str) -> dict:
    """
    Load an image file, validate it, optimize if needed, and encode to base64.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary containing:
            - 'encoded_data': Base64-encoded image string
            - 'format': Image format (e.g., 'jpeg', 'png')
            - 'size_mb': Final size in megabytes
            - 'original_size_mb': Original size in megabytes
            - 'reduction_applied': Boolean indicating if size reduction was applied
            - 'reduction_techniques': List of techniques applied (if any)
    
    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If image format is not supported or size cannot be reduced
    """
    # Validate image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file format (PNG, JPEG, JPG)
    file_extension = os.path.splitext(image_path)[1].lower()
    supported_formats = ['.png', '.jpg', '.jpeg']
    
    if file_extension not in supported_formats:
        raise ValueError(
            f"Unsupported image format: {file_extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    # Load image with Pillow
    try:
        image = Image.open(image_path)
        image.load()  # Ensure image is fully loaded
    except FileNotFoundError:
        # Re-raise with more context
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except PermissionError:
        raise PermissionError(
            f"Permission denied: Cannot read image file '{image_path}'. "
            "Please check your file permissions."
        )
    except Image.UnidentifiedImageError:
        raise ValueError(
            f"Cannot identify image file '{image_path}'. "
            f"The file may be corrupted or not a valid image. "
            f"Supported formats: PNG, JPEG, JPG"
        )
    except OSError as e:
        raise OSError(f"Failed to open image file '{image_path}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load image '{image_path}': {str(e)}")
    
    # Determine image format
    image_format = image.format if image.format else 'JPEG'
    original_size_mb = get_image_size_mb(image, image_format)
    
    # Call size reduction if needed (> 5MB)
    reduction_applied = False
    reduction_techniques = []
    
    if original_size_mb > 3.5:
        print(f"Image size ({original_size_mb:.2f}MB) exceeds 3.5MB limit. Applying size reduction...")
        try:
            image, reduction_techniques = reduce_image_size(image, target_size_mb=3.5)
            reduction_applied = True
            print(f"Size reduction successful. Techniques applied: {', '.join(reduction_techniques)}")
        except ValueError as e:
            # Provide clear error message for size reduction failure
            raise ValueError(
                f"Image size reduction failed: {str(e)}\n"
                f"Original size: {original_size_mb:.2f}MB (exceeds 3.5MB limit)\n"
                f"Suggestion: Manually resize or compress the image before uploading."
            )
    
    # Get final size
    final_size_mb = get_image_size_mb(image, image_format)
    
    # Base64 encode image
    try:
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()
        encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image to base64: {str(e)}")
    
    # Check base64-encoded size (base64 increases size by ~33%)
    encoded_size_mb = len(encoded_data) / (1024 * 1024)
    
    # If encoded size exceeds 5MB, we need to reduce further
    # Target 3.5MB for image file to ensure <5MB after base64 encoding
    if encoded_size_mb > 5.0:
        try:
            image, additional_techniques = reduce_image_size(image, target_size_mb=3.5)
            reduction_applied = True
            reduction_techniques.extend(additional_techniques)
            
            # Re-encode with new size
            buffer = BytesIO()
            image.save(buffer, format=image_format)
            image_bytes = buffer.getvalue()
            encoded_data = base64.b64encode(image_bytes).decode('utf-8')
            
            final_size_mb = get_image_size_mb(image, image_format)
            encoded_size_mb = len(encoded_data) / (1024 * 1024)
            
        except Exception as e:
            raise ValueError(
                f"Image base64-encoded size exceeds 5MB limit: {encoded_size_mb:.2f}MB\n"
                f"Suggestion: Manually resize or compress the image before uploading."
            )
    
    # Return dict with encoded data and metadata
    return {
        'encoded_data': encoded_data,
        'format': image_format.lower(),
        'size_mb': final_size_mb,
        'encoded_size_mb': encoded_size_mb,
        'original_size_mb': original_size_mb,
        'reduction_applied': reduction_applied,
        'reduction_techniques': reduction_techniques
    }


def create_bedrock_client(profile: str = None, region: str = 'us-east-1'):
    """
    Create a boto3 Bedrock Runtime client with proper credential handling.
    
    Args:
        profile: AWS CLI profile name (optional, uses default if not specified)
        region: AWS region (default: 'us-east-1')
    
    Returns:
        boto3.client: Configured Bedrock Runtime client
    
    Raises:
        NoCredentialsError: If AWS credentials are not configured
        ClientError: If there's an error creating the client
        BotoCoreError: For other boto3-related errors
    """
    try:
        # Create session with profile if specified
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        
        # Create Bedrock Runtime client
        client = session.client('bedrock-runtime')
        
        return client
        
    except NoCredentialsError as e:
        # Provide clear setup instructions for missing credentials
        raise NoCredentialsError(
            "AWS credentials not found. Please configure your credentials using one of these methods:\n"
            "  1. Run 'aws configure' to set up default credentials\n"
            "  2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables\n"
            "  3. Use --profile argument to specify an AWS CLI profile\n"
            "\n"
            "For more information, visit: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html"
        )
    except ClientError as e:
        # Handle AWS API errors with helpful messages
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        # Provide context-specific error messages
        if error_code == 'InvalidClientTokenId':
            helpful_message = (
                f"Invalid AWS credentials. The access key ID you provided does not exist.\n"
                f"Please verify your credentials are correct and try again."
            )
        elif error_code == 'SignatureDoesNotMatch':
            helpful_message = (
                f"Invalid AWS credentials. The secret access key is incorrect.\n"
                f"Please verify your credentials and try again."
            )
        elif error_code == 'ExpiredToken':
            helpful_message = (
                f"AWS credentials have expired. Please refresh your credentials and try again."
            )
        else:
            helpful_message = f"Failed to create Bedrock client: {error_message}"
        
        raise ClientError(
            {
                'Error': {
                    'Code': error_code,
                    'Message': helpful_message
                }
            },
            'CreateBedrockClient'
        )
    except BotoCoreError as e:
        # Handle network and connection issues
        error_message = str(e)
        
        # Provide helpful context for common network issues
        if 'could not connect' in error_message.lower() or 'connection' in error_message.lower():
            helpful_message = (
                f"Network connection error: Unable to connect to AWS Bedrock service.\n"
                f"Please check your internet connection and try again.\n"
                f"Original error: {error_message}"
            )
        elif 'timeout' in error_message.lower():
            helpful_message = (
                f"Network timeout: The request to AWS Bedrock timed out.\n"
                f"Please check your internet connection and try again.\n"
                f"Original error: {error_message}"
            )
        elif 'endpoint' in error_message.lower():
            helpful_message = (
                f"Endpoint resolution error: Unable to resolve AWS Bedrock endpoint.\n"
                f"Please verify the region '{region}' is correct and supports Bedrock.\n"
                f"Original error: {error_message}"
            )
        else:
            helpful_message = (
                f"AWS connection error: {error_message}\n"
                f"This may be due to network issues or service configuration problems."
            )
        
        raise BotoCoreError(helpful_message)
    except Exception as e:
        # Handle profile not found or other session errors
        error_message = str(e)
        
        if 'could not be found' in error_message.lower() or 'profile' in error_message.lower():
            raise ValueError(
                f"AWS profile '{profile}' not found.\n"
                f"Available profiles are listed in ~/.aws/credentials\n"
                f"Please check your AWS configuration or use a different profile."
            )
        raise Exception(f"Unexpected error creating Bedrock client: {error_message}")


def create_vertex_client(project_id: str, region: str = 'us-central1', model_id: str = 'gemini-1.5-pro'):
    """
    Create a Google Cloud Gemini client with proper credential handling.
    Uses the google-genai package with Application Default Credentials (ADC).
    
    Args:
        project_id: GCP project ID
        region: GCP region (default: 'us-central1')
        model_id: Gemini model ID (default: 'gemini-1.5-pro')
    
    Returns:
        tuple: (client, model_id) - Configured Gemini client and model ID
    
    Raises:
        ImportError: If google-genai is not installed
        ValueError: If project_id is invalid or credentials are missing
        Exception: For other GCP-related errors
    """
    try:
        # Import the new google-genai package
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required for Gemini integration. "
            "Install it with: pip install google-genai"
        )
    
    # Validate project_id
    if not project_id or not project_id.strip():
        raise ValueError(
            "GCP project ID is required when using Gemini. "
            "Please provide a valid project ID using the --gcp-project argument."
        )
    
    try:
        # Create client with ADC (Application Default Credentials)
        # This automatically uses credentials from gcloud auth application-default login
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region
        )
        
        return (client, model_id)
        
    except Exception as e:
        error_message = str(e)
        
        # Handle credential errors gracefully
        if 'credentials' in error_message.lower() or 'authentication' in error_message.lower() or 'default credentials' in error_message.lower():
            raise ValueError(
                "GCP credentials not found or invalid. Please configure your credentials using one of these methods:\n"
                "  1. Run 'gcloud auth application-default login' to set up Application Default Credentials (ADC)\n"
                "  2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to a service account key file\n"
                "  3. Use a service account when running on GCP (Compute Engine, Cloud Run, etc.)\n"
                "\n"
                "For more information, visit: https://cloud.google.com/docs/authentication/getting-started"
            )
        elif 'project' in error_message.lower():
            raise ValueError(
                f"Invalid GCP project ID '{project_id}'. "
                "Please verify the project ID is correct and you have access to it."
            )
        elif 'permission' in error_message.lower() or 'access' in error_message.lower():
            raise ValueError(
                f"Access denied to GCP project '{project_id}' or Vertex AI API. "
                "Please verify:\n"
                "  1. The project ID is correct\n"
                "  2. Vertex AI API is enabled for this project\n"
                "  3. Your credentials have the necessary permissions (e.g., 'Vertex AI User' role)"
            )
        else:
            raise Exception(f"Failed to create Gemini client: {error_message}")


def invoke_bedrock_model(client, model_id: str, image_data_list: list, prompt: str) -> dict:
    """
    Invoke AWS Bedrock model with one or more images and a prompt using the Converse API.
    
    The Converse API provides a unified, model-agnostic interface that works across
    all Bedrock models (Claude, Llama, Mistral, Amazon Nova, etc.) without needing
    model-specific payload formats.
    
    Args:
        client: boto3 Bedrock Runtime client
        model_id: Bedrock model ID (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0')
        image_data_list: List of dictionaries, each containing 'encoded_data' and 'format' keys
        prompt: Text prompt/question about the image(s)
    
    Returns:
        Dictionary containing:
            - 'transcription': The model's response text
            - 'model_id': The model ID used
            - 'input_tokens': Number of input tokens used (if available)
            - 'output_tokens': Number of output tokens used (if available)
            - 'stop_reason': Reason the model stopped generating (if available)
    
    Raises:
        ClientError: If there's an error invoking the Bedrock API
        ValueError: If the model ID is invalid or response parsing fails
        BotoCoreError: For network or connection issues
    """
    try:
        import json

        # Build content blocks for the Converse API
        content = []

        # Add all images as ImageBlock entries
        for image_data in image_data_list:
            fmt = image_data['format']
            # Converse expects format values: png, jpeg, gif, webp
            if fmt == 'jpg':
                fmt = 'jpeg'
            content.append({
                'image': {
                    'format': fmt,
                    'source': {
                        'bytes': base64.b64decode(image_data['encoded_data'])
                    }
                }
            })

        # Add the text prompt
        content.append({'text': prompt})

        # Capture timestamp before invocation
        invocation_timestamp = datetime.now(timezone.utc).isoformat()

        # Invoke Bedrock Converse API
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ],
            inferenceConfig={
                'maxTokens': 4096
            }
        )

        # Extract transcription from the unified response format
        output_message = response.get('output', {}).get('message', {})
        output_content = output_message.get('content', [])

        text_blocks = [block['text'] for block in output_content if 'text' in block]
        if not text_blocks:
            raise ValueError("No content found in model response")
        transcription = '\n'.join(text_blocks)

        # Extract usage metadata from the standardized usage object
        usage = response.get('usage', {})
        input_tokens = usage.get('inputTokens')
        output_tokens = usage.get('outputTokens')

        # Extract stop reason
        stop_reason = response.get('stopReason')

        return {
            'transcription': transcription,
            'model_id': model_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'stop_reason': stop_reason,
            'timestamp': invocation_timestamp
        }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))

        # Provide helpful error messages for common issues
        if error_code == 'ValidationException':
            helpful = f"Invalid model ID or request format: {error_message}"
        elif error_code == 'ResourceNotFoundException':
            helpful = f"Model not found: {model_id}. Please check the model ID and ensure you have access to it."
        elif error_code == 'ThrottlingException':
            helpful = f"Request throttled. Please try again later or reduce request rate."
        elif error_code == 'AccessDeniedException':
            helpful = f"Access denied to model {model_id}. Please check your IAM permissions."
        else:
            helpful = f"Bedrock API error: {error_message}"

        raise ClientError(
            {'Error': {'Code': error_code, 'Message': helpful}},
            'Converse'
        )

    except BotoCoreError as e:
        raise BotoCoreError(
            f"Network or connection error while invoking Bedrock model: {str(e)}"
        )

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse Bedrock response: {str(e)}")

    except Exception as e:
        raise Exception(f"Unexpected error invoking Bedrock model: {str(e)}")


def invoke_vertex_model(client_and_model: tuple, image_data_list: list, prompt: str) -> dict:
    """
    Invoke Google Cloud Gemini model with one or more images and a prompt.
    Uses the google-genai package.
    
    Args:
        client_and_model: Tuple of (client, model_id) from create_vertex_client()
        image_data_list: List of dictionaries, each containing 'encoded_data' and 'format' keys
        prompt: Text prompt/question about the image(s)
    
    Returns:
        Dictionary containing:
            - 'transcription': The model's response text
            - 'model_id': The model ID used
            - 'input_tokens': Number of input tokens used (if available)
            - 'output_tokens': Number of output tokens used (if available)
            - 'stop_reason': Reason the model stopped generating (if available)
            - 'timestamp': ISO 8601 timestamp of invocation
    
    Raises:
        ValueError: If response parsing fails or model returns no content
        Exception: For GCP-specific API errors
    """
    try:
        # Import required modules
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required for Gemini integration. "
            "Install it with: pip install google-genai"
        )
    
    # Unpack client and model_id
    client, model_id = client_and_model
    
    try:
        # Capture timestamp before invocation
        invocation_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Construct request with image(s) and prompt
        # Convert base64 images to Part objects
        content_parts = []
        
        for image_data in image_data_list:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data['encoded_data'])
            
            # Determine MIME type from format
            mime_type = f"image/{image_data['format']}"
            if image_data['format'] == 'jpg':
                mime_type = "image/jpeg"
            
            # Create Part object from image bytes
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            content_parts.append(image_part)
        
        # Add text prompt as the last part
        content_parts.append(prompt)
        
        # Invoke Gemini model with generate_content()
        response = client.models.generate_content(
            model=model_id,
            contents=content_parts
        )
        
        # Parse response and extract text
        if not response or not response.text:
            raise ValueError("No content found in Gemini model response")
        
        transcription = response.text
        
        # Extract metadata (consistent with Bedrock format)
        input_tokens = None
        output_tokens = None
        stop_reason = None
        
        # Try to extract token usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                input_tokens = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                output_tokens = usage.candidates_token_count
        
        # Try to extract finish reason if available
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                # Convert Gemini finish reason to Bedrock-style stop reason
                finish_reason = str(candidate.finish_reason)
                if 'STOP' in finish_reason:
                    stop_reason = 'end_turn'
                elif 'MAX_TOKENS' in finish_reason:
                    stop_reason = 'max_tokens'
                else:
                    stop_reason = finish_reason.lower()
        
        return {
            'transcription': transcription,
            'model_id': model_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'stop_reason': stop_reason,
            'timestamp': invocation_timestamp
        }
        
    except Exception as e:
        error_message = str(e)
        
        # Handle common GCP API errors
        if 'permission' in error_message.lower() or 'denied' in error_message.lower():
            raise ValueError(
                f"Access denied to Gemini API. Please verify:\n"
                f"  1. Vertex AI API is enabled for your project\n"
                f"  2. Your credentials have the necessary permissions (e.g., 'Vertex AI User' role)\n"
                f"  3. The project ID is correct\n"
                f"Original error: {error_message}"
            )
        elif 'not found' in error_message.lower() or '404' in error_message:
            raise ValueError(
                f"Resource not found in Gemini. This may indicate:\n"
                f"  1. Invalid model name: {model_id}\n"
                f"  2. Model not available in the specified region\n"
                f"  3. Invalid project ID\n"
                f"Original error: {error_message}"
            )
        elif 'quota' in error_message.lower() or 'rate limit' in error_message.lower() or '429' in error_message:
            raise ValueError(
                f"Gemini quota exceeded or rate limit reached. Please try again later.\n"
                f"Original error: {error_message}"
            )
        elif 'invalid' in error_message.lower() or '400' in error_message:
            raise ValueError(
                f"Invalid request to Gemini. This may indicate:\n"
                f"  1. Invalid image format or size\n"
                f"  2. Invalid prompt or parameters\n"
                f"  3. Model doesn't support the requested operation\n"
                f"Original error: {error_message}"
            )
        else:
            raise Exception(f"Unexpected error invoking Gemini model: {error_message}")
    """
    Invoke Google Cloud Vertex AI Gemini model with one or more images and a prompt.
    
    Args:
        model: vertexai.generative_models.GenerativeModel instance
        image_data_list: List of dictionaries, each containing 'encoded_data' and 'format' keys
        prompt: Text prompt/question about the image(s)
    
    Returns:
        Dictionary containing:
            - 'transcription': The model's response text
            - 'model_id': The model ID used
            - 'input_tokens': Number of input tokens used (if available)
            - 'output_tokens': Number of output tokens used (if available)
            - 'stop_reason': Reason the model stopped generating (if available)
            - 'timestamp': ISO 8601 timestamp of invocation
    
    Raises:
        ValueError: If response parsing fails or model returns no content
        Exception: For GCP-specific API errors
    """
    try:
        # Import required Vertex AI modules
        from vertexai.generative_models import Part
        import google.api_core.exceptions
    except ImportError:
        raise ImportError(
            "google-cloud-aiplatform is required for Vertex AI integration. "
            "Install it with: pip install google-cloud-aiplatform>=1.38.0"
        )
    
    try:
        # Capture timestamp before invocation
        invocation_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Construct Vertex AI request with image(s) and prompt
        # Convert base64 images to Part objects
        content_parts = []
        
        for image_data in image_data_list:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data['encoded_data'])
            
            # Determine MIME type from format
            mime_type = f"image/{image_data['format']}"
            if image_data['format'] == 'jpg':
                mime_type = "image/jpeg"
            
            # Create Part object from image bytes
            image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
            content_parts.append(image_part)
        
        # Add text prompt as the last part
        content_parts.append(prompt)
        
        # Invoke Gemini model with generate_content()
        response = model.generate_content(content_parts)
        
        # Parse response and extract text
        if not response or not response.text:
            raise ValueError("No content found in Vertex AI model response")
        
        transcription = response.text
        
        # Extract metadata (consistent with Bedrock format)
        # Vertex AI provides usage metadata in response
        input_tokens = None
        output_tokens = None
        stop_reason = None
        
        # Try to extract token usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                input_tokens = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                output_tokens = usage.candidates_token_count
        
        # Try to extract finish reason if available
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                # Convert Vertex AI finish reason to Bedrock-style stop reason
                finish_reason = str(candidate.finish_reason)
                if 'STOP' in finish_reason:
                    stop_reason = 'end_turn'
                elif 'MAX_TOKENS' in finish_reason:
                    stop_reason = 'max_tokens'
                else:
                    stop_reason = finish_reason.lower()
        
        # Get model name from the model object
        model_id = model._model_name if hasattr(model, '_model_name') else 'gemini-1.5-pro'
        
        return {
            'transcription': transcription,
            'model_id': model_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'stop_reason': stop_reason,
            'timestamp': invocation_timestamp
        }
        
    except google.api_core.exceptions.PermissionDenied as e:
        raise ValueError(
            f"Access denied to Vertex AI API. Please verify:\n"
            f"  1. Vertex AI API is enabled for your project\n"
            f"  2. Your credentials have the necessary permissions (e.g., 'Vertex AI User' role)\n"
            f"  3. The project ID is correct\n"
            f"Original error: {str(e)}"
        )
    
    except google.api_core.exceptions.NotFound as e:
        raise ValueError(
            f"Resource not found in Vertex AI. This may indicate:\n"
            f"  1. Invalid model name\n"
            f"  2. Model not available in the specified region\n"
            f"  3. Invalid project ID\n"
            f"Original error: {str(e)}"
        )
    
    except google.api_core.exceptions.ResourceExhausted as e:
        raise ValueError(
            f"Vertex AI quota exceeded or rate limit reached. Please try again later.\n"
            f"Original error: {str(e)}"
        )
    
    except google.api_core.exceptions.InvalidArgument as e:
        raise ValueError(
            f"Invalid request to Vertex AI. This may indicate:\n"
            f"  1. Invalid image format or size\n"
            f"  2. Invalid prompt or parameters\n"
            f"  3. Model doesn't support the requested operation\n"
            f"Original error: {str(e)}"
        )
    
    except google.api_core.exceptions.GoogleAPIError as e:
        raise Exception(
            f"Google Cloud API error: {str(e)}\n"
            f"Please check your GCP configuration and try again."
        )
    
    except ValueError:
        # Re-raise ValueError as-is
        raise
    
    except Exception as e:
        raise Exception(f"Unexpected error invoking Vertex AI model: {str(e)}")


def format_output(response: dict, model_id: str, image_data_list: list = None, reduction_info: list = None, accuracy_metrics: dict = None) -> str:
    """
    Format the transcription response for display.
    
    Args:
        response: Dictionary containing transcription and metadata from invoke_bedrock_model
        model_id: The Bedrock model ID used
        image_data_list: Optional list of dictionaries containing image metadata
        reduction_info: Optional list of dictionaries with reduction information per image
        accuracy_metrics: Optional dictionary containing accuracy metrics (cer, wer, similarity_score)
    
    Returns:
        Formatted string ready for console display or file output
    """
    output_lines = []
    
    # Header
    output_lines.append("="*80)
    output_lines.append("TRANSCRIPTION RESULT")
    output_lines.append("="*80)
    output_lines.append("")
    
    # Model ID
    output_lines.append(f"Model: {model_id}")
    
    # Number of images processed
    if image_data_list:
        output_lines.append(f"Images processed: {len(image_data_list)}")
    
    # Timestamp
    if response.get('timestamp'):
        output_lines.append(f"Timestamp: {response['timestamp']}")
    
    # Token usage if available
    if response.get('input_tokens') is not None and response.get('output_tokens') is not None:
        output_lines.append(f"Tokens: {response['input_tokens']} input, {response['output_tokens']} output")
    
    # Stop reason if available
    if response.get('stop_reason'):
        output_lines.append(f"Stop reason: {response['stop_reason']}")
    
    # Size reduction info if applied to any images
    if reduction_info and len(reduction_info) > 0:
        output_lines.append("")
        output_lines.append("Image Size Reduction Applied:")
        for info in reduction_info:
            output_lines.append(f"  {info['path']}: {', '.join(info['techniques'])}")
    
    # Accuracy metrics if available
    if accuracy_metrics:
        output_lines.append("")
        output_lines.append("Accuracy Metrics:")
        output_lines.append(f"  Character Error Rate (CER): {accuracy_metrics['cer']:.4f} (lower is better)")
        output_lines.append(f"  Word Error Rate (WER): {accuracy_metrics['wer']:.4f} (lower is better)")
        output_lines.append(f"  Similarity Score: {accuracy_metrics['similarity_score']:.4f} (higher is better)")
    
    # Transcription text
    output_lines.append("")
    output_lines.append("Transcription:")
    output_lines.append("-"*80)
    output_lines.append(response['transcription'])
    output_lines.append("")
    output_lines.append("="*80)
    
    return "\n".join(output_lines)


def save_to_file(output: str, filepath: str) -> None:
    """
    Save formatted output to a file.
    
    Args:
        output: The formatted output string to save
        filepath: Path where the file should be saved
    
    Raises:
        OSError: If there's an error creating directories or writing the file
        PermissionError: If there's no permission to write to the file or directory
        ValueError: If the filepath is invalid
    """
    # Validate filepath
    if not filepath or not filepath.strip():
        raise ValueError("Output filepath cannot be empty")
    
    try:
        # Create directory if needed
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except PermissionError:
                raise PermissionError(
                    f"Permission denied: Cannot create directory '{output_dir}'. "
                    "Please check your permissions or choose a different location."
                )
            except OSError as e:
                raise OSError(
                    f"Failed to create directory '{output_dir}': {str(e)}"
                )
        
        # Write formatted output to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(output)
        except PermissionError:
            raise PermissionError(
                f"Permission denied: Cannot write to file '{filepath}'. "
                "Please check your permissions or choose a different location."
            )
        except OSError as e:
            raise OSError(
                f"Failed to write to file '{filepath}': {str(e)}"
            )
    except (PermissionError, OSError, ValueError):
        # Re-raise these specific exceptions
        raise
    except Exception as e:
        raise OSError(f"Unexpected error saving file '{filepath}': {str(e)}")


def extract_text_from_pdf(pdf_path: str, skip_first_page: bool = True) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        skip_first_page: If True, skip the first page (default: True for ground truth PDFs)
    
    Returns:
        Extracted text as a string (concatenated from all pages)
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF parsing fails or file is invalid
        PermissionError: If there's no permission to read the file
    """
    # Import PyMuPDF (fitz)
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF text extraction. "
            "Install it with: pip install PyMuPDF>=1.23.0"
        )
    
    # Validate PDF path exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Open PDF file with PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        
        # Determine starting page (skip first page if requested)
        start_page = 1 if skip_first_page else 0
        
        # Iterate through pages and extract text
        extracted_text = []
        for page_num in range(start_page, len(doc)):
            page = doc[page_num]
            text = page.get_text()
            extracted_text.append(text)
        
        # Close the document
        doc.close()
        
        # Concatenate all text
        full_text = '\n'.join(extracted_text)
        
        return full_text
        
    except PermissionError:
        raise PermissionError(
            f"Permission denied: Cannot read PDF file '{pdf_path}'. "
            "Please check your file permissions."
        )
    except fitz.FileDataError as e:
        raise ValueError(
            f"Invalid or corrupted PDF file '{pdf_path}': {str(e)}"
        )
    except fitz.EmptyFileError:
        raise ValueError(
            f"Empty PDF file '{pdf_path}'. The file contains no data."
        )
    except Exception as e:
        raise ValueError(
            f"Failed to extract text from PDF '{pdf_path}': {str(e)}"
        )


def calculate_accuracy(predicted: str, ground_truth: str) -> dict:
    """
    Calculate accuracy metrics between predicted and ground truth text.
    
    Args:
        predicted: The predicted/transcribed text
        ground_truth: The ground truth text to compare against
    
    Returns:
        Dictionary containing:
            - 'cer': Character Error Rate (0 = perfect match, lower is better)
            - 'wer': Word Error Rate (0 = perfect match, lower is better)
            - 'similarity_score': Similarity score (1 = perfect match, higher is better)
    """
    import difflib
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        raise ImportError(
            "rapidfuzz is required for accuracy calculation. "
            "Install it with: pip install rapidfuzz>=3.0.0"
        )
    
    # Normalize both text strings (lowercase, whitespace)
    def normalize_text(text: str) -> str:
        """Normalize text by converting to lowercase, collapsing whitespace, and removing brackets/carets."""
        import re
        # Convert to lowercase
        text = text.lower()
        # Remove brackets and their contents: [illegible], [?], etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Remove angle brackets and their contents: <unclear>, etc.
        text = re.sub(r'<.*?>', '', text)
        # Remove standalone carets
        text = text.replace('^', '')
        # Collapse multiple whitespaces to single space
        text = ' '.join(text.split())
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    normalized_predicted = normalize_text(predicted)
    
    # Remove the first line of ground truth (document title, not part of transcription)
    ground_truth_lines = ground_truth.split('\n')
    ground_truth_no_title = '\n'.join(ground_truth_lines[1:])
    
    # Strip trailing "Keywords: ..." section if present (not part of the original image)
    import re
    ground_truth_no_title = re.split(r'\nKeywords:?', ground_truth_no_title, maxsplit=1)[0]
    
    normalized_ground_truth = normalize_text(ground_truth_no_title)
    
    # Calculate Character Error Rate (CER) using rapidfuzz
    # CER = edit_distance / len(ground_truth)
    if len(normalized_ground_truth) == 0:
        # Handle edge case: empty ground truth
        cer = 0.0 if len(normalized_predicted) == 0 else 1.0
    else:
        char_edit_distance = Levenshtein.distance(normalized_predicted, normalized_ground_truth)
        cer = char_edit_distance / len(normalized_ground_truth)
    
    # Calculate Word Error Rate (WER)
    # WER = (substitutions + deletions + insertions) / total_words_in_ground_truth
    predicted_words = normalized_predicted.split()
    ground_truth_words = normalized_ground_truth.split()
    
    if len(ground_truth_words) == 0:
        # Handle edge case: empty ground truth
        wer = 0.0 if len(predicted_words) == 0 else 1.0
    else:
        word_edit_distance = Levenshtein.distance(predicted_words, ground_truth_words)
        wer = word_edit_distance / len(ground_truth_words)
    
    # Calculate similarity score using rapidfuzz.fuzz.ratio()
    # Returns a value between 0 and 100, normalized to 0-1 (1 = perfect match)
    # This is more forgiving than SequenceMatcher for text with structural differences
    similarity_score = fuzz.ratio(normalized_predicted, normalized_ground_truth) / 100.0
    
    return {
        'cer': cer,
        'wer': wer,
        'similarity_score': similarity_score,
        'normalized_predicted': normalized_predicted,
        'normalized_ground_truth': normalized_ground_truth
    }


def log_to_dynamodb(table_name: str, experiment_data: dict, region: str = 'us-east-1', profile: str = None) -> bool:
    """
    Log experiment data to DynamoDB table.
    
    Args:
        table_name: Name of the DynamoDB table
        experiment_data: Dictionary containing experiment data with keys:
            - image_path: Path to the image file
            - provider: Cloud provider used ('aws' or 'gcp')
            - model_id: Model ID used (Bedrock or Gemini)
            - prompt: The prompt/question sent to the model
            - reduction_techniques: List of image size reduction techniques applied
            - output_path: Path where results were saved (nullable)
            - accuracy: Accuracy score if ground truth was provided (nullable)
            - character_error_rate: CER metric if ground truth was provided (nullable)
            - word_error_rate: WER metric if ground truth was provided (nullable)
            - ground_truth_path: Path to ground truth PDF (nullable)
            - input_tokens: Number of input tokens used (nullable)
            - output_tokens: Number of output tokens generated (nullable)
        region: AWS region (default: 'us-east-1')
        profile: AWS CLI profile name (optional)
    
    Returns:
        Boolean indicating success (True) or failure (False)
    """
    try:
        # Create DynamoDB client with boto3
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        
        dynamodb = session.client('dynamodb')
        
        # Generate unique experiment ID (UUID)
        experiment_id = str(uuid.uuid4())
        
        # Add current timestamp (ISO 8601 format)
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Construct item with all required fields
        item = {
            'experiment_id': {'S': experiment_id},
            'timestamp': {'S': timestamp},
            'image_path': {'S': experiment_data.get('image_path', '')},
            'provider': {'S': experiment_data.get('provider', 'aws')},
            'model_id': {'S': experiment_data.get('model_id', '')},
            'prompt': {'S': experiment_data.get('prompt', '')},
        }
        
        # Add reduction_techniques as a list
        reduction_techniques = experiment_data.get('reduction_techniques', [])
        if reduction_techniques:
            item['reduction_techniques'] = {
                'L': [{'S': technique} for technique in reduction_techniques]
            }
        else:
            item['reduction_techniques'] = {'L': []}
        
        # Add nullable fields (only if they have values)
        output_path = experiment_data.get('output_path')
        if output_path:
            item['output_path'] = {'S': output_path}
        
        accuracy = experiment_data.get('accuracy')
        if accuracy is not None:
            item['accuracy'] = {'N': str(accuracy)}
        
        character_error_rate = experiment_data.get('character_error_rate')
        if character_error_rate is not None:
            item['character_error_rate'] = {'N': str(character_error_rate)}
        
        word_error_rate = experiment_data.get('word_error_rate')
        if word_error_rate is not None:
            item['word_error_rate'] = {'N': str(word_error_rate)}
        
        ground_truth_path = experiment_data.get('ground_truth_path')
        if ground_truth_path:
            item['ground_truth_path'] = {'S': ground_truth_path}
        
        input_tokens = experiment_data.get('input_tokens')
        if input_tokens is not None:
            item['input_tokens'] = {'N': str(input_tokens)}
        
        output_tokens = experiment_data.get('output_tokens')
        if output_tokens is not None:
            item['output_tokens'] = {'N': str(output_tokens)}
        
        # Write item to DynamoDB table
        dynamodb.put_item(
            TableName=table_name,
            Item=item
        )
        
        print(f"✓ Experiment logged to DynamoDB (ID: {experiment_id})")
        return True
        
    except NoCredentialsError:
        print("⚠ DynamoDB logging failed: AWS credentials not found")
        print("  Experiment data was not logged, but transcription completed successfully")
        return False
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'ResourceNotFoundException':
            print(f"⚠ DynamoDB logging failed: Table '{table_name}' not found")
            print("  Please create the table before using --dynamodb-table option")
        elif error_code == 'AccessDeniedException':
            print(f"⚠ DynamoDB logging failed: Access denied to table '{table_name}'")
            print("  Please check your IAM permissions")
        else:
            print(f"⚠ DynamoDB logging failed: {error_message}")
        
        print("  Experiment data was not logged, but transcription completed successfully")
        return False
    
    except BotoCoreError as e:
        print(f"⚠ DynamoDB logging failed: Network error - {str(e)}")
        print("  Experiment data was not logged, but transcription completed successfully")
        return False
    
    except Exception as e:
        print(f"⚠ DynamoDB logging failed: {str(e)}")
        print("  Experiment data was not logged, but transcription completed successfully")
        return False


def main():
    """Main entry point for the Bedrock Image Analyzer."""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Send images to AWS Bedrock models for transcription and analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python image_transcriber.py image.jpg
  
  # Multiple images in one request
  python image_transcriber.py image1.jpg image2.jpg image3.jpg
  
  # With options
  python image_transcriber.py page1.png page2.png --model anthropic.claude-3-haiku-20240307-v1:0
  python image_transcriber.py *.png --prompt "Transcribe all text from these images"
  python image_transcriber.py image1.jpg image2.jpg --output results.txt --profile myprofile
        """
    )
    
    # Required argument: image_paths (now accepts multiple)
    parser.add_argument(
        'image_paths',
        type=str,
        nargs='+',
        help='Path(s) to the image file(s) to analyze (can specify multiple images)'
    )
    
    # Optional argument: --model
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model ID to use (default: anthropic.claude-3-sonnet-20240229-v1:0 for AWS, gemini-1.5-pro for GCP)'
    )
    
    # Optional argument: --prompt
    parser.add_argument(
        '--prompt',
        type=str,
        default='Transcribe and describe this image',
        help='Custom prompt or question about the image (default: "Transcribe and describe this image")'
    )
    
    # Optional argument: --prompt-file
    parser.add_argument(
        '--prompt-file',
        type=str,
        default=None,
        help='Read prompt from a text file instead of command line (optional)'
    )
    
    # Optional argument: --output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to the specified file (optional)'
    )
    
    # Optional argument: --profile
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='AWS CLI profile name to use (optional, uses default credentials if not specified)'
    )
    
    # Optional argument: --provider
    parser.add_argument(
        '--provider',
        type=str,
        default='aws',
        choices=['aws', 'gcp'],
        help='Cloud provider to use: aws (AWS Bedrock) or gcp (Google Cloud Vertex AI) (default: aws)'
    )
    
    # Optional argument: --gcp-project
    parser.add_argument(
        '--gcp-project',
        type=str,
        default=None,
        help='GCP project ID (required when --provider is gcp)'
    )
    
    # Optional argument: --region
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help='Region to use (default: us-east-1 for AWS, us-east1 for GCP)'
    )
    
    # Optional argument: --ground-truth
    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='Path to ground truth PDF file for accuracy evaluation (optional)'
    )
    
    # Optional argument: --dynamodb-table
    parser.add_argument(
        '--dynamodb-table',
        type=str,
        default=None,
        help='DynamoDB table name for logging experiments (optional)'
    )
    
    # Optional argument: --dynamodb-region
    parser.add_argument(
        '--dynamodb-region',
        type=str,
        default='us-east-1',
        help='AWS region for DynamoDB table (default: us-east-1, independent of --region)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set model default based on provider if not specified
    if args.model is None:
        if args.provider == 'aws':
            args.model = 'anthropic.claude-3-sonnet-20240229-v1:0'
        elif args.provider == 'gcp':
            args.model = 'gemini-1.5-pro'
    
    # Set region default based on provider if not specified
    if args.region is None:
        if args.provider == 'aws':
            args.region = 'us-east-1'
        elif args.provider == 'gcp':
            args.region = 'us-east1'
    
    # Validate that --gcp-project is provided when --provider is 'gcp'
    if args.provider == 'gcp' and not args.gcp_project:
        parser.error("--gcp-project is required when --provider is 'gcp'")
    
    try:
        # Load prompt from file if --prompt-file is specified
        if args.prompt_file:
            print(f"Loading prompt from file: {args.prompt_file}")
            try:
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    args.prompt = f.read().strip()
                print(f"Prompt loaded successfully ({len(args.prompt)} characters)")
            except FileNotFoundError:
                print(f"\nError: Prompt file not found: {args.prompt_file}")
                return 1
            except Exception as e:
                print(f"\nError reading prompt file: {e}")
                return 1
        
        # Load and encode all images
        print(f"Loading {len(args.image_paths)} image(s)...")
        image_data_list = []
        all_reduction_info = []
        
        for idx, image_path in enumerate(args.image_paths, 1):
            print(f"\n[Image {idx}/{len(args.image_paths)}] {image_path}")
            image_data = load_and_encode_image(image_path)
            image_data_list.append(image_data)
            
            print(f"  Format: {image_data['format']}")
            print(f"  Original size: {image_data['original_size_mb']:.2f}MB")
            print(f"  Final size: {image_data['size_mb']:.2f}MB")
            print(f"  Base64-encoded size: {image_data['encoded_size_mb']:.2f}MB")
            
            if image_data['reduction_applied']:
                print(f"  Size reduction applied")
                print(f"  Techniques used: {', '.join(image_data['reduction_techniques'])}")
                all_reduction_info.append({
                    'path': image_path,
                    'techniques': image_data['reduction_techniques']
                })
        
        print(f"\n✓ All {len(image_data_list)} image(s) loaded successfully")
        
        # Branch logic based on --provider argument
        if args.provider == 'aws':
            # Create AWS Bedrock client
            print(f"\nConnecting to AWS Bedrock (region: {args.region})...")
            if args.profile:
                print(f"Using AWS profile: {args.profile}")
            
            try:
                client = create_bedrock_client(profile=args.profile, region=args.region)
                print("Connected successfully")
            except (NoCredentialsError, ClientError, BotoCoreError) as e:
                # Handle AWS-specific errors
                print(f"\nAWS Connection Error: {e}")
                return 1
            
            # Invoke AWS Bedrock model
            print(f"\nInvoking AWS Bedrock model: {args.model}")
            print(f"Prompt: {args.prompt}")
            print(f"Sending {len(image_data_list)} image(s) to model...")
            
            try:
                response = invoke_bedrock_model(
                    client=client,
                    model_id=args.model,
                    image_data_list=image_data_list,
                    prompt=args.prompt
                )
            except (ClientError, BotoCoreError, ValueError) as e:
                # Handle AWS Bedrock invocation errors
                print(f"\nAWS Bedrock Invocation Error: {e}")
                return 1
        
        elif args.provider == 'gcp':
            # Create Google Cloud Vertex AI client
            print(f"\nConnecting to Google Cloud Vertex AI (region: {args.region})...")
            print(f"GCP Project: {args.gcp_project}")
            
            try:
                client_and_model = create_vertex_client(
                    project_id=args.gcp_project,
                    region=args.region,
                    model_id=args.model
                )
                print("Connected successfully")
            except (ValueError, ImportError) as e:
                # Handle GCP-specific errors
                print(f"\nGCP Connection Error: {e}")
                return 1
            except Exception as e:
                # Handle unexpected GCP errors
                print(f"\nGCP Connection Error: {e}")
                return 1
            
            # Invoke Google Cloud Vertex AI model
            print(f"\nInvoking Google Cloud Vertex AI model: {args.model}")
            print(f"Prompt: {args.prompt}")
            print(f"Sending {len(image_data_list)} image(s) to model...")
            
            try:
                response = invoke_vertex_model(
                    client_and_model=client_and_model,
                    image_data_list=image_data_list,
                    prompt=args.prompt
                )
            except (ValueError, ImportError) as e:
                # Handle GCP Vertex AI invocation errors
                print(f"\nGCP Vertex AI Invocation Error: {e}")
                return 1
            except Exception as e:
                # Handle unexpected GCP invocation errors
                print(f"\nGCP Vertex AI Invocation Error: {e}")
                return 1
        
        else:
            # This should never happen due to argparse choices validation
            print(f"\nError: Unsupported provider '{args.provider}'. Supported providers: aws, gcp")
            return 1
        
        # Calculate accuracy if ground truth is provided
        accuracy_metrics = None
        if args.ground_truth:
            print(f"\nCalculating accuracy against ground truth...")
            print(f"Ground truth file: {args.ground_truth}")
            
            try:
                # Extract text from ground truth PDF
                ground_truth_text = extract_text_from_pdf(args.ground_truth)
                print(f"Ground truth extracted: {len(ground_truth_text)} characters")
                
                # Calculate accuracy metrics
                accuracy_metrics = calculate_accuracy(response['transcription'], ground_truth_text)
                print(f"✓ Accuracy metrics calculated")
                
                # Display normalized texts for comparison
                print(f"\n{'='*80}")
                print("NORMALIZED TEXT COMPARISON")
                print('='*80)
                print(f"\nModel Transcription (normalized, {len(accuracy_metrics['normalized_predicted'])} chars):")
                print('-'*80)
                print(accuracy_metrics['normalized_predicted'][:500] + ('...' if len(accuracy_metrics['normalized_predicted']) > 500 else ''))
                print(f"\nGround Truth (normalized, {len(accuracy_metrics['normalized_ground_truth'])} chars):")
                print('-'*80)
                print(accuracy_metrics['normalized_ground_truth'][:500] + ('...' if len(accuracy_metrics['normalized_ground_truth']) > 500 else ''))
                print('='*80)
                
            except FileNotFoundError:
                print(f"⚠ Error: Ground truth file not found: {args.ground_truth}")
            except Exception as e:
                print(f"⚠ Error calculating accuracy: {e}")
        
        # Format and display output (includes accuracy metrics if available)
        formatted_output = format_output(response, args.model, image_data_list, all_reduction_info, accuracy_metrics)
        print(f"\n{formatted_output}")
        
        # Log to DynamoDB if table is specified
        if args.dynamodb_table:
            print(f"\n{'='*80}")
            print("DynamoDB Logging")
            print('='*80)
            print(f"Table: {args.dynamodb_table}")
            
            # Collect all experiment data
            experiment_data = {
                'image_path': ', '.join(args.image_paths),  # Join multiple image paths
                'provider': args.provider,  # Cloud provider ('aws' or 'gcp')
                'model_id': args.model,
                'prompt': args.prompt,
                'reduction_techniques': [],
                'output_path': args.output,
                'accuracy': None,
                'character_error_rate': None,
                'word_error_rate': None,
                'ground_truth_path': args.ground_truth,
                'input_tokens': response.get('input_tokens'),
                'output_tokens': response.get('output_tokens')
            }
            
            # Collect all reduction techniques from all images
            for reduction_info in all_reduction_info:
                experiment_data['reduction_techniques'].extend(reduction_info['techniques'])
            
            # Add accuracy metrics if available
            if accuracy_metrics:
                experiment_data['accuracy'] = accuracy_metrics['similarity_score']
                experiment_data['character_error_rate'] = accuracy_metrics['cer']
                experiment_data['word_error_rate'] = accuracy_metrics['wer']
            
            # Call log_to_dynamodb() with experiment data
            # Note: DynamoDB is AWS-only, uses separate region from provider region
            print("Logging experiment data...")
            log_to_dynamodb(
                table_name=args.dynamodb_table,
                experiment_data=experiment_data,
                region=args.dynamodb_region,
                profile=args.profile
            )
        
        # Save to file if requested
        if args.output:
            print(f"\nSaving results to: {args.output}")
            
            # Format output for file (include additional metadata)
            file_output_lines = []
            file_output_lines.append("Bedrock Image Transcription Result")
            file_output_lines.append("="*80)
            file_output_lines.append("")
            file_output_lines.append(f"Images ({len(args.image_paths)}):")
            for img_path in args.image_paths:
                file_output_lines.append(f"  - {img_path}")
            file_output_lines.append(f"Prompt: {args.prompt}")
            if args.ground_truth:
                file_output_lines.append(f"Ground truth: {args.ground_truth}")
            file_output_lines.append("")
            
            # Add the formatted output (without the header since we have our own)
            formatted_lines = formatted_output.split('\n')
            # Skip the first 4 lines (header) and add the rest
            file_output_lines.extend(formatted_lines[4:])
            
            output_content = "\n".join(file_output_lines)
            
            # Use the save_to_file function
            save_to_file(output_content, args.output)
            
            print(f"Results saved successfully")
        
        print("\nDone!")
        
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}")
        return 1
    
    except PermissionError as e:
        print(f"\nPermission Error: {e}")
        return 1
    
    except OSError as e:
        print(f"\nFile System Error: {e}")
        return 1
    
    except ValueError as e:
        print(f"\nValidation Error: {e}")
        return 1
    
    except NoCredentialsError as e:
        print(f"\nAWS Credentials Error:")
        print(str(e))
        return 1
    
    except ClientError as e:
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"\nAWS API Error: {error_message}")
        return 1
    
    except BotoCoreError as e:
        print(f"\nAWS Connection Error: {e}")
        return 1
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
