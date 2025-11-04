import os
import json
import requests
from typing import Optional
import logging
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv, find_dotenv

# Load .env from repository root so environment variables are available without manual export
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


def _env_first(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _resolve_api_settings_for_deployment(deployment: Optional[str] = None):
    """Return (api_key, api_base, api_type, api_version) for a given deployment.

    Supports per-deployment keys and versions via env vars:
    - AZURE_OPENAI_API_KEY_GPT4O, AZURE_OPENAI_API_KEY_EMBED, AZURE_OPENAI_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_VERSION_GPT4O, AZURE_OPENAI_API_VERSION
    - AZURE_OPENAI_API_KEY as generic fallback
    """
    endpoint = _env_first('AZURE_OPENAI_ENDPOINT', 'OPENAI_API_BASE', 'OPENAI_API_BASE')

    # Decide which key to use based on deployment hints
    key = None
    if deployment:
        # If deployment matches known GPT4O deployment env, prefer that key
        gpt4o_dep = os.environ.get('GPT4O_DEPLOYMENT') or os.environ.get('GPT4O_DEPLOYMENT_NAME') or os.environ.get('GPT4O_MODEL_NAME')
        if gpt4o_dep and deployment == gpt4o_dep:
            key = _env_first('AZURE_OPENAI_API_KEY_GPT4O')
    # If not determined, prefer embed key for embeddings
    if not key:
        # If deployment name suggests an embedding model, prefer embed key
        if deployment and ('embed' in deployment.lower() or 'embed' in (os.environ.get('AZURE_OPENAI_EMBED_MODEL') or '').lower()):
            key = _env_first('AZURE_OPENAI_API_KEY_EMBED')
    if not key:
        # generic fallbacks
        key = _env_first('AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_KEY', 'OPENAI_API_KEY')

    api_type = os.environ.get('OPENAI_API_TYPE') or ('azure' if endpoint else None)

    # Per-deployment API version override
    api_version = None
    if deployment:
        # try specific env like AZURE_OPENAI_API_VERSION_GPT4O if deployment env name present
        dep_env_suffix = None
        if 'gpt4' in (deployment.lower() if deployment else '') or 'gpt-4' in (deployment.lower() if deployment else ''):
            dep_env_suffix = 'GPT4O'
        if dep_env_suffix:
            api_version = os.environ.get(f'AZURE_OPENAI_API_VERSION_{dep_env_suffix}')
    if not api_version:
        api_version = _env_first('AZURE_OPENAI_API_VERSION', 'OPENAI_API_VERSION', 'AZURE_OPENAI_API_VERSION')
    if not api_version:
        api_version = '2023-10-01'

    return key, endpoint, api_type, api_version


def _create_openai_client_for_deployment(deployment: Optional[str] = None) -> OpenAI:
    key, endpoint, api_type, api_version = _resolve_api_settings_for_deployment(deployment)
    # Masked debug info
    try:
        if key:
            logger.info(f"Using API key for deployment {deployment or ''}: {key[:4]}...{key[-4:]}")
        else:
            logger.info(f"No specific API key found for deployment {deployment or ''}; client will use environment or fail")
        if endpoint:
            logger.info(f"Using API base: {endpoint}")
    except Exception:
        pass

    # Build explicit kwargs for AzureOpenAI client
    client_kwargs = {}
    if key:
        client_kwargs['api_key'] = key
    if endpoint:
        client_kwargs['azure_endpoint'] = endpoint
    if api_version:
        client_kwargs['api_version'] = api_version

    try:
        # Construct AzureOpenAI client explicitly with the resolved credentials
        client = AzureOpenAI(**client_kwargs)
        return client
    except Exception:
        logger.exception('Failed to create AzureOpenAI client with explicit credentials; falling back to default')
        # Fallback: return a client that reads from environment
        return AzureOpenAI()


def _get_deployment(deployment_param: Optional[str] = None) -> Optional[str]:
    """Resolve deployment/engine name: prefer explicit param, then AZURE_OPENAI_DEPLOYMENT env var, then GPT4O_DEPLOYMENT."""
    return deployment_param or os.environ.get('AZURE_OPENAI_DEPLOYMENT') or os.environ.get('GPT4O_DEPLOYMENT') or os.environ.get('GPT4O_DEPLOYMENT_NAME')


def chat_with_model(deployment: Optional[str], user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, images_b64: Optional[list] = None) -> str:
    """Call Azure/OpenAI ChatCompletion using openai>=1.0.0 client.

    Returns the assistant content string.
    """
    dep = _get_deployment(deployment)
    if not dep:
        logger.error('No Azure OpenAI deployment provided. Set AZURE_OPENAI_DEPLOYMENT or pass deployment param.')
        raise RuntimeError('No Azure OpenAI deployment configured')

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Build user content
    user_content = []
    if images_b64:
        for b64 in images_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    user_content.append({"type": "text", "text": user_prompt})
    
    messages.append({"role": "user", "content": user_content})

    client = _create_openai_client_for_deployment(dep)
    try:
        resp = client.chat.completions.create(model=dep, messages=messages, temperature=temperature)
        choice = resp.choices[0]
        # New SDK: choice.message.content
        try:
            return choice.message.content
        except Exception:
            return str(choice)
    except Exception:
        logger.exception('Azure OpenAI chat call failed')
        raise


def call_generate_api(model: str, prompt: str, images_b64: Optional[list] = None, api_url: str = "http://localhost:11434/api/generate") -> str:
    """Call Ollama generate API for text or multimodal prompts.

    - model: model name (e.g., 'llava')
    - prompt: the text prompt
    - images_b64: list of base64-encoded images
    - api_url: Ollama API endpoint

    Returns the response text or empty string on failure.
    """
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if images_b64:
            payload["images"] = images_b64

        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception:
        logger.exception('Ollama generate call failed')
        return ''


# Ensure a usable OPENAI_API_KEY and OPENAI_API_BASE are present for the new OpenAI client
def _inject_openai_fallback_env():
    """Inject Azure OpenAI env vars to ensure fallback client works."""
    if not os.environ.get('AZURE_OPENAI_API_KEY'):
        azure_key = _env_first('AZURE_OPENAI_API_KEY_GPT4O', 'AZURE_OPENAI_API_KEY_EMBED', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_KEY')
        if azure_key:
            os.environ['AZURE_OPENAI_API_KEY'] = azure_key
            logger.info('Injected AZURE_OPENAI_API_KEY from Azure vars')
    if not os.environ.get('AZURE_OPENAI_ENDPOINT'):
        endpoint = _env_first('AZURE_OPENAI_ENDPOINT', 'OPENAI_API_BASE')
        if endpoint:
            os.environ['AZURE_OPENAI_ENDPOINT'] = endpoint
            logger.info('Injected AZURE_OPENAI_ENDPOINT from Azure vars')
    if not os.environ.get('OPENAI_API_VERSION'):
        version = _env_first('AZURE_OPENAI_API_VERSION_GPT4O', 'AZURE_OPENAI_API_VERSION')
        if version:
            os.environ['OPENAI_API_VERSION'] = version
            logger.info('Injected OPENAI_API_VERSION from Azure vars')
    if not os.environ.get('OPENAI_API_TYPE'):
        os.environ['OPENAI_API_TYPE'] = 'azure'
        logger.info('Injected OPENAI_API_TYPE=azure')


# Call it at import time to set env vars early
_inject_openai_fallback_env()
