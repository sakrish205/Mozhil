"""
Explanation Generator Module
Template-based explanation system (Ollama-free).
Supports Tamil, English, Hindi, Malayalam, and Telugu.
"""

from typing import Dict, Any

# Language-specific explanation templates
EXPLANATION_TEMPLATES = {
    "English": {
        "AI_GENERATED": [
            "This audio sample is classified as AI-generated with {confidence:.1%} confidence. The analysis detected {artifacts}. These characteristics are typically associated with synthetic speech synthesis.",
            "High probability of AI generation ({confidence:.1%}) based on {artifacts}. The speech patterns exhibit unnatural consistency and reduced micro-variations.",
            "Synthetic voice detected with {confidence:.1%} confidence. Key indicators include {artifacts} and artificial spectral characteristics."
        ],
        "HUMAN": [
            "This audio sample is classified as human-generated with {confidence:.1%} confidence. The analysis detected {artifacts}. These characteristics are consistent with natural human speech patterns.",
            "Natural human speech detected ({confidence:.1%}) with key indicators like {artifacts}. The audio exhibits organic pitch variations and natural formant transitions.",
            "Human voice confirmed with {confidence:.1%} confidence. The presence of {artifacts} aligns with organic vocal cord characteristics."
        ]
    },
    "Tamil": {
        "AI_GENERATED": "இந்த ஆடியோ மாதிரி AI-ஆல் உருவாக்கப்பட்டது ({confidence:.1%} உறுதி). {artifacts} கண்டறியப்பட்டுள்ளன. இது செயற்கை பேச்சு தொகுப்புடன் தொடர்புடையது.",
        "HUMAN": "இந்த ஆடியோ மாதிரி ஒரு மனிதரால் பேசப்பட்டது ({confidence:.1%} உறுதி). {artifacts} கண்டறியப்பட்டுள்ளன. இது இயற்கையான மனித பேச்சு முறைகளுடன் ஒத்துப்போகிறது."
    },
    "Hindi": {
        "AI_GENERATED": "यह ऑडियो नमूना AI-जनित के रूप में वर्गीकृत है ({confidence:.1%} आत्मविश्वास)। {artifacts} का पता चला है। ये विशेषताएं कृत्रिम वाक् संश्लेषण से जुड़ी हैं।",
        "HUMAN": "यह ऑडियो नमूना मानव-जनित के रूप में वर्गीकृत है ({confidence:.1%} आत्मविश्वास)। {artifacts} का पता चला है। ये विशेषताएं प्राकृतिक मानव भाषण पैटर्न के अनुरूप हैं।"
    },
    "Malayalam": {
        "AI_GENERATED": "ഈ ഓഡിയോ സാമ്പിൾ AI-നിർമ്മിതമായി തരംതിരിച്ചിരിക്കുന്നു ({confidence:.1%} ആത്മവിശ്വാസം). {artifacts} കണ്ടെത്തി. ഈ സവിശേഷതകൾ സിന്തറ്റിക് സ്പീച്ച് സിന്തസിസുമായി ബന്ധപ്പെട്ടതാണ്.",
        "HUMAN": "ഈ ഓഡിയോ സാമ്പിൾ മനുഷ്യൻ സംസാരിച്ചതായി തരംതിരിച്ചിരിക്കുന്നു ({confidence:.1%} ആത്മവിശ്വാസം). {artifacts} കണ്ടെത്തി. ഈ സവിശേഷതകൾ സ്വാഭാവിക മനുഷ്യ സംസാര രീതികളുമായി പൊരുത്തപ്പെടുന്നു."
    },
    "Telugu": {
        "AI_GENERATED": "ఈ ఆడియో నమూనా AI-ద్వారా రూపొందించబడినట్లుగా వర్గీకరించబడింది ({confidence:.1%} విశ్వాసం). {artifacts} గుర్తించబడ్డాయి. ఈ లక్షణాలు కృత్రిమ ప్రసంగ సంశ్లేషణతో సంబంధం కలిగి ఉంటాయి.",
        "HUMAN": "ఈ ఆడియో నమూనా మానవ-జనితమైనదిగా వర్గీకరించబడింది ({confidence:.1%} విశ్వాసం). {artifacts} గుర్తించబడ్డాయి. ఈ లక్షణాలు సహజ మానవ ప్రసంగ నమూనాలకు అనుగుణంగా ఉన్నాయి."
    }
}

# Artifact translations for non-English templates
ARTIFACT_NAMES = {
    "unnaturally consistent pitch patterns": {
        "Tamil": "இயற்கைக்கு மாறான சுருதி முறைகள்",
        "Hindi": "अस्वाभाविक रूप से सुसंगत पिच पैटर्न",
        "Malayalam": "അസ്വാഭാവികമായ പിച്ച് പാറ്റേണുകൾ",
        "Telugu": "అసాధారణమైన పిచ్ నమూనాలు"
    },
    "synthetic spectral envelope": {
        "Tamil": "செயற்கை நிறமாலை உறை",
        "Hindi": "कृत्रिम वर्णक्रमीय लिफाफा",
        "Malayalam": "സിന്തറ്റിക് സ്പെക്ട്രൽ എൻവലപ്പ്",
        "Telugu": "కృత్రిమ స్పెక్ట్రల్ ఎన్వలప్"
    },
    "reduced micro-variations in formants": {
        "Tamil": "குறைக்கப்பட்ட நுண் மாறுபாடுகள்",
        "Hindi": "फॉर्मेंट्स में कम सूक्ष्म बदलाव",
        "Malayalam": "കുറഞ്ഞ സൂക്ഷ്മ വ്യതിയാനങ്ങൾ",
        "Telugu": "తగ్గిన మైక్రో-వైవిధ్యాలు"
    },
    "natural breathing patterns": {
        "Tamil": "இயற்கையான சுவாச முறைகள்",
        "Hindi": "प्राकृतिक सांस लेने के पैटर्न",
        "Malayalam": "സ്വാഭാവിക ശ്വസന രീതികൾ",
        "Telugu": "సహజ శ్వాస నమూనాలు"
    },
    "organic pitch variations": {
        "Tamil": "இயற்கையான சுருதி மாறுபாடுகள்",
        "Hindi": "प्राकृतिक पिच विविधताएं",
        "Malayalam": "സ്വാഭാവിക പിച്ച് വ്യത്യാസങ്ങൾ",
        "Telugu": "సహజ పిచ్ వైవిధ్యాలు"
    },
    "natural formant transitions": {
        "Tamil": "இயற்கையான பேச்சொலி மாற்றங்கள்",
        "Hindi": "प्राकृतिक फॉर्मेंट संक्रमण",
        "Malayalam": "സ്വാഭാവിക ഫോർമാന്റ് സംക്രമണങ്ങൾ",
        "Telugu": "సహజమైన ఫార్మెంట్ పరిವರ್తనాలు"
    }
}


class ExplanationGenerator:
    """
    Generate explanations for voice classification results using templates.
    No Ollama or external LLM dependency.
    """
    
    def __init__(self):
        # Always available as it's template-based
        self.available_model = "TemplateSystem"
    
    def generate_explanation(
        self,
        classification: str,
        confidence: float,
        metadata: Dict[str, Any],
        target_language: str = "English"
    ) -> str:
        """
        Produce an explanation using localized templates.
        
        Args:
            classification: "AI_GENERATED" or "HUMAN"
            confidence: Confidence score (0-1)
            metadata: Analysis metadata from classifier
            target_language: Target language name
            
        Returns:
            Localized explanation string
        """
        # Normalize language name
        lang = target_language if target_language in EXPLANATION_TEMPLATES else "English"
        
        # Get indicators and artifacts
        indicators = metadata.get("analysis_indicators", {})
        artifacts_list = indicators.get("detected_artifacts", [])
        
        # Format artifacts for the specific language
        if lang == "English":
            artifacts_text = ", ".join(artifacts_list) if artifacts_list else "specific audio markers"
        else:
            # Try to translate artifact names, otherwise use them as is
            translated_artifacts = []
            for art in artifacts_list:
                translated = ARTIFACT_NAMES.get(art, {}).get(lang, art)
                translated_artifacts.append(translated)
            artifacts_text = ", ".join(translated_artifacts) if translated_artifacts else "ஆடியோ குறிகாட்டிகள்" if lang=="Tamil" else "ऑडियो संकेतक" if lang=="Hindi" else "ഓഡിയോ മാർക്കറുകൾ" if lang=="Malayalam" else "ఆడియో గుర్తులు"
            
        # Select template
        template = EXPLANATION_TEMPLATES[lang][classification]
        
        # If English, we have a list of alternatives to keep it varied
        if isinstance(template, list):
            import random
            template = random.choice(template)
            
        return template.format(
            confidence=confidence,
            artifacts=artifacts_text
        )


# Singleton instance
explanation_generator = ExplanationGenerator()
