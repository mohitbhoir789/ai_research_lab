#!/usr/bin/env python3
"""
Test script to verify Gemini integration in the LLM handler.
"""

import asyncio
import logging
from backend.app.utils.llm import LLMHandler, LLMConfig, LLMProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_gemini():
    """Test Gemini integration in the LLM handler."""
    logger.info("Testing Gemini integration...")
    
    # Initialize LLM handler
    llm_handler = LLMHandler()
    
    # Create Gemini configuration
    gemini_config = LLMConfig(
        model="gemini-1.5-flash",  # Using the flash model for quick testing
        provider=LLMProvider.GEMINI,
        temperature=0.7,
        max_tokens=100
    )
    
    # Test prompt
    prompt = "Explain briefly what makes LLMs like Gemini useful for research."
    
    try:
        # Generate text using Gemini
        logger.info(f"Sending prompt to Gemini: {prompt}")
        response, metadata = await llm_handler.generate(prompt=prompt, config=gemini_config)
        
        # Print results
        logger.info(f"Gemini response received successfully!")
        logger.info(f"Provider: {metadata.get('provider')}")
        logger.info(f"Model: {metadata.get('model')}")
        logger.info(f"Response: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Gemini: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Gemini integration in LLM handler...")
    success = asyncio.run(test_gemini())
    
    if success:
        print("\nGemini test completed successfully! ✅")
    else:
        print("\nGemini test failed! ❌")