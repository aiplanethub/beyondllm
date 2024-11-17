from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass, field
import json
import boto3
import botocore

@dataclass
class BedrockModel:
    """
    Class representing AWS Bedrock service for accessing various LLMs
    
    Authentication Methods (in order of precedence):
    1. Explicitly provided credentials in constructor
    2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    3. AWS credentials file (~/.aws/credentials)
    4. IAM role (if running on AWS)
    
    Example:
    from beyondllm.llms import BedrockModel
    
    # Method 1: Using environment variables (recommended)
    # First set these environment variables:
    # export AWS_ACCESS_KEY_ID="your_access_key"
    # export AWS_SECRET_ACCESS_KEY="your_secret_key"
    # export AWS_DEFAULT_REGION="us-east-1"
    llm = BedrockModel(model_id="anthropic.claude-instant-v1")
    
    # Method 2: Using AWS profile
    llm = BedrockModel(
        model_id="anthropic.claude-instant-v1",
        profile_name="my-aws-profile"
    )
    
    # Method 3: Explicit credentials (not recommended for production)
    llm = BedrockModel(
        model_id="anthropic.claude-instant-v1",
        aws_access_key_id="your_access_key",
        aws_secret_access_key="your_secret_key",
        region="us-east-1"
    )
    """
    model_id: str = field(default="anthropic.claude-3-haiku-20240307-v1:0")
    region: str = field(default="")
    model_kwargs: dict = field(default_factory=lambda: {
        "max_tokens_to_sample": 500,
        "temperature": 0,
        "top_p": 1,
    })
    aws_access_key_id: str = field(default="")
    aws_secret_access_key: str = field(default="")
    profile_name: str = field(default="")
    
    def __post_init__(self):
        # Set region from environment if not provided
        if not self.region:
            self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        # Set credentials from environment if not provided
        if not self.aws_access_key_id:
            self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        if not self.aws_secret_access_key:
            self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
            
        self.load_llm()
    
    def _get_bedrock_client(self):
        """
        Creates and returns a boto3 client for Amazon Bedrock Runtime
        """
        try:
            session_kwargs = {"region_name": self.region}
            
            # If profile is specified, use it
            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name
            # If explicit credentials are provided, use them
            elif self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            
            # Create session with retry configuration
            retry_config = botocore.config.Config(
                region_name=self.region,
                retries={
                    "max_attempts": 10,
                    "mode": "standard"
                }
            )
            
            session = boto3.Session(**session_kwargs)
            
            # Create the Bedrock Runtime client
            bedrock_client = session.client(
                service_name='bedrock-runtime',
                config=retry_config
            )
            
            
            
            return bedrock_client
            
        except botocore.exceptions.ProfileNotFound:
            raise ValueError(f"AWS profile '{self.profile_name}' not found in credentials file")
        except botocore.exceptions.NoCredentialsError:
            raise ValueError(
                "No AWS credentials found. Please provide credentials using one of these methods:\n"
                "1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "2. AWS credentials file (~/.aws/credentials)\n"
                "3. IAM role (if running on AWS)\n"
                "4. Explicit credentials in constructor"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Bedrock client: {str(e)}")
    
    def load_llm(self):
        """
        Initialize the Bedrock client
        """
        self.client = self._get_bedrock_client()
        return self.client
    
    def _format_prompt(self, prompt: str) -> Dict:
        """
        Format the prompt based on the model provider
        """
        if self.model_id.startswith("anthropic."):
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.model_kwargs.get("max_tokens_to_sample", 500),
                "temperature": self.model_kwargs.get("temperature", 0),
                "top_p": self.model_kwargs.get("top_p", 1),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            }
        
        elif self.model_id.startswith("meta."):
            return{
                "prompt": prompt
            }
        elif self.model_id.startswith("amazon."):
            return {
            "inputText": prompt,
            
            }
        
        elif self.model_id.startswith("ai21."):
            return {
                "prompt": prompt,
                "maxTokens": self.model_kwargs.get("max_tokens_to_sample", 200),
                "temperature": self.model_kwargs.get("temperature", 0.5),
                "topP": self.model_kwargs.get("top_p", 0.5)
            }
        elif self.model_id.startswith("mistral."):
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        else:
            raise ValueError(f"Unsupported model provider in model_id: {self.model_id}")
    
    def _parse_response(self, response: Dict) -> str:
        """
        Parse the response based on the model provider
        """
        response_body = json.loads(response.get("body").read())
        
        if self.model_id.startswith("anthropic."):
        # Extract the text content from the response
            messages = response_body.get("content", [])
            for message in messages:
                if message.get("type") == "text":
                    return message.get("text", "")
            return ""
        elif self.model_id.startswith("amazon."):
            return response_body.get("results", [{}])[0].get("outputText", "")
        elif self.model_id.startswith("ai21."):
            return response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
        
        elif self.model_id.startswith("meta."):
            return response_body.get("generation")
        elif self.model_id.startswith("ai21."):
       
            completions = response_body.get("completions", [])
            if completions and "data" in completions[0]:
                return completions[0]["data"].get("text", "")
            return ""
        
        elif self.model_id.startswith("mistral."):
        # Mistral models respond with a body containing the formatted messages
            return response_body.get("messages", [{}])[0].get("content", "")
       
        else:
            raise ValueError(f"Unsupported model provider in model_id: {self.model_id}")
    
    def predict(self, prompt: Any) -> str:
        """
        Generate prediction using the Bedrock model
        
        Parameters:
            prompt (Any): Input prompt for generation
            
        Returns:
            str: Generated text response
        """
        try:
            # Format the request body based on the model provider
            body = self._format_prompt(prompt)
            
            # Invoke the model
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse and return the response
            return self._parse_response(response)
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")
    
    @staticmethod
    def load_from_kwargs(self, kwargs):
        """
        Load model configuration from kwargs
        """
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()