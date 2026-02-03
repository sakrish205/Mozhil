"""
Explanation Generator Module
Uses Ollama with translategemma for multi-language explanations.
"""

import requests
import json
from typing import Dict, Any, Optional

# Language mappings
LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu"
}

LANGUAGE_CODES = {
    "english": "en",
    "tamil": "ta",
    "hindi": "hi",
    "malayalam": "ml",
    "telugu": "te"
}


class ExplanationGenerator:
    """
    Generate explanations for voice classification results.
    Uses Ollama with translategemma for multi-language support.
    """
    
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "translategemma:4b"
    FALLBACK_MODEL = "llama3.2:3b"  # Smaller fallback if translategemma unavailable
    
    def __init__(self):
        self.available_model = self._check_available_model()
    
    def _check_available_model(self) -> Optional[str]:
        """Check which model is available in Ollama."""
        try:
            response = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check for translategemma first
                for name in model_names:
                    if "translategemma" in name.lower():
                        return name
                
                # Check for fallback
                for name in model_names:
                    if any(x in name.lower() for x in ["llama", "gemma", "mistral", "phi"]):
                        return name
                
                if model_names:
                    return model_names[0]
            
            return None
        except Exception as e:
            print(f"Error checking Ollama: {e}")
            return None
    
    def _call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """Make a request to Ollama API."""
        model = model or self.available_model or self.MODEL_NAME
        
        try:
            response = requests.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 256
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Ollama error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Make sure it's running.")
            return ""
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def generate_explanation(
        self,
        classification: str,
        confidence: float,
        metadata: Dict[str, Any],
        target_language: str = "en"
    ) -> str:
        """
        Generate a detailed explanation for the classification result.
        
        Args:
            classification: "AI" or "Human"
            confidence: Confidence score (0-1)
            metadata: Analysis metadata from classifier
            target_language: Target language code (en, ta, hi, ml, te)
            
        Returns:
            Explanation string in target language
        """
        # Build explanation prompt
        indicators = metadata.get("analysis_indicators", {})
        artifacts = indicators.get("detected_artifacts", [])
        
        artifacts_text = ", ".join(artifacts) if artifacts else "no specific artifacts"
        
        prompt = f"""Generate a brief, technical explanation for a voice authenticity analysis result.

Classification: {classification}-generated voice
Confidence: {confidence:.1%}
Pattern Type: {indicators.get('pattern_consistency', 'unknown')}
Spectral Analysis: {indicators.get('spectral_characteristics', 'unknown')}
Key Findings: {artifacts_text}

Write a 2-3 sentence explanation that:
1. States the classification clearly
2. Mentions the confidence level
3. Explains the key audio characteristics that led to this conclusion

Be concise and technical. Do not use markdown or special formatting."""

        # Get base explanation in English
        explanation = self._call_ollama(prompt)
        
        if not explanation:
            # Fallback to template-based explanation
            explanation = self._generate_fallback_explanation(
                classification, confidence, indicators
            )
        
        # Translate if needed
        if target_language != "en" and target_language in LANGUAGE_NAMES:
            explanation = self._translate(explanation, target_language)
        
        return explanation.strip()
    
    def _generate_fallback_explanation(
        self,
        classification: str,
        confidence: float,
        indicators: Dict[str, Any]
    ) -> str:
        """Generate explanation without LLM (fallback)."""
        artifacts = indicators.get("detected_artifacts", [])
        
        if classification == "AI":
            base = f"This audio sample is classified as AI-generated with {confidence:.1%} confidence."
            if artifacts:
                base += f" The analysis detected {', '.join(artifacts[:2])}."
            base += " These characteristics are typically associated with synthetic speech synthesis."
        else:
            base = f"This audio sample is classified as human-generated with {confidence:.1%} confidence."
            if artifacts:
                base += f" The analysis detected {', '.join(artifacts[:2])}."
            base += " These characteristics are consistent with natural human speech patterns."
        
        return base
    
    def _translate(self, text: str, target_language: str) -> str:
        """Translate text to target language using Ollama."""
        lang_name = LANGUAGE_NAMES.get(target_language, target_language)
        
        # Use translategemma specifically for translation
        prompt = f"""Translate the following English text to {lang_name}. 
Only output the translation, nothing else.

Text: {text}

Translation:"""
        
        translated = self._call_ollama(prompt, self.MODEL_NAME)
        
        # Return original if translation fails
        return translated if translated else text


# Singleton instance
explanation_generator = ExplanationGenerator()
