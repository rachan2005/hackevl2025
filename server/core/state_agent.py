
"""
State Agent for ADK-based interview system
Processes behavioral features and correlates them with conversation timestamps
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .behavioral_extractors import BehavioralExtractorFactory

logger = logging.getLogger(__name__)

@dataclass
class BehavioralFeature:
    """Represents a behavioral feature extracted from external systems"""
    timestamp: float
    feature_type: str  # e.g., "emotion", "sentiment", "attention", "stress"
    value: Any
    confidence: float
    description: str

@dataclass
class EnrichedQAPair:
    """Represents a complete Q&A pair with behavioral insights"""
    question: str
    answer: str
    timestamp: float
    behavioral_features: List[BehavioralFeature] = field(default_factory=list)
    behavioral_insights: str = ""
    confidence: float = 0.0
    is_enriched: bool = False
    last_updated: float = 0.0

@dataclass
class CorrelatedInsight:
    """Represents a correlated insight between conversation and behavioral features"""
    conversation_timestamp: float
    conversation_type: str  # "question" or "answer"
    conversation_content: str
    behavioral_features: List[BehavioralFeature]
    contextual_description: str
    correlation_confidence: float

class StateAgent:
    """Processes state information and correlates behavioral features with conversation"""
    
    def __init__(self, session_id: str, shared_state: Dict[str, Any], extractor_type: str = "dummy", extractor_config: Dict[str, Any] = None):
        self.session_id = session_id
        self.shared_state = shared_state
        self.behavioral_features: List[BehavioralFeature] = []
        self.correlated_insights: List[CorrelatedInsight] = []
        self.enriched_qa_pairs: List[EnrichedQAPair] = []
        self.is_processing = False
        self.websocket = None  # Will be set by the websocket handler
        
        # Initialize behavioral extractor
        if extractor_config is None:
            extractor_config = {}
        self.behavioral_extractor = BehavioralExtractorFactory.create_extractor(extractor_type, extractor_config)
        logger.info(f"StateAgent initialized with {extractor_type} behavioral extractor")
        
    async def process_signal(self, signal: Dict[str, Any]):
        """Process signals from the conversational agent"""
        signal_type = signal.get("type")
        data = signal.get("data", {})
        timestamp = signal.get("timestamp", time.time())
        
        logger.info(f"🎯 State agent received signal: {signal_type}")
        logger.info(f"📊 Signal data: {data}")
        
        if signal_type == "question_timestamped":
            await self._process_question_timestamp(data, timestamp)
        elif signal_type == "answer_timestamped":
            await self._process_answer_timestamp(data, timestamp)
        elif signal_type == "qa_pair_timestamped":
            await self._process_qa_pair_timestamp(data, timestamp)
        else:
            logger.warning(f"⚠️ Unknown signal type: {signal_type}")
    
    async def _process_question_timestamp(self, data: Dict[str, Any], signal_timestamp: float):
        """Process a question timestamp and correlate with behavioral features"""
        question = data.get("question", "")
        question_timestamp = data.get("timestamp", signal_timestamp)
        
        logger.info(f"Processing question timestamp: {question_timestamp}")
        
        # Find behavioral features around this timestamp
        relevant_features = self._find_features_around_timestamp(question_timestamp)
        
        # Generate contextual description
        contextual_description = self._generate_contextual_description(
            "question", question, relevant_features
        )
        
        # Create correlated insight
        insight = CorrelatedInsight(
            conversation_timestamp=question_timestamp,
            conversation_type="question",
            conversation_content=question,
            behavioral_features=relevant_features,
            contextual_description=contextual_description,
            correlation_confidence=self._calculate_correlation_confidence(relevant_features)
        )
        
        self.correlated_insights.append(insight)
        
        # Update shared state with enriched information
        self.shared_state["question_insights"] = self._get_question_insights()
        self.shared_state["last_question_insight"] = insight
        self.shared_state["last_update"] = time.time()
        
        logger.info(f"Generated question insight: {contextual_description}")
    
    async def _process_answer_timestamp(self, data: Dict[str, Any], signal_timestamp: float):
        """Process an answer timestamp and correlate with behavioral features"""
        answer = data.get("answer", "")
        answer_timestamp = data.get("timestamp", signal_timestamp)
        
        logger.info(f"Processing answer timestamp: {answer_timestamp}")
        
        # Find behavioral features around this timestamp
        relevant_features = self._find_features_around_timestamp(answer_timestamp)
        
        # Generate contextual description
        contextual_description = self._generate_contextual_description(
            "answer", answer, relevant_features
        )
        
        # Create correlated insight
        insight = CorrelatedInsight(
            conversation_timestamp=answer_timestamp,
            conversation_type="answer",
            conversation_content=answer,
            behavioral_features=relevant_features,
            contextual_description=contextual_description,
            correlation_confidence=self._calculate_correlation_confidence(relevant_features)
        )
        
        self.correlated_insights.append(insight)
        
        # Update shared state with enriched information
        self.shared_state["answer_insights"] = self._get_answer_insights()
        self.shared_state["last_answer_insight"] = insight
        self.shared_state["last_update"] = time.time()
        
        logger.info(f"Generated answer insight: {contextual_description}")
    
    async def _process_qa_pair_timestamp(self, data: Dict[str, Any], signal_timestamp: float):
        """Process a complete Q&A pair and create unified enriched structure"""
        question = data.get("question", "")
        answer = data.get("answer", "")
        qa_timestamp = data.get("timestamp", signal_timestamp)
        
        logger.info(f"🔄 Processing Q&A pair: {question[:50]}...")
        
        # Create initial enriched Q&A pair (without behavioral insights yet)
        enriched_qa = EnrichedQAPair(
            question=question,
            answer=answer,
            timestamp=qa_timestamp,
            last_updated=time.time()
        )
        
        # Add to enriched Q&A pairs list
        self.enriched_qa_pairs.append(enriched_qa)
        
        # Update shared state with the new Q&A pair
        self.shared_state["enriched_qa_pairs"] = self._get_enriched_qa_pairs_dict()
        self.shared_state["last_qa_pair"] = enriched_qa
        self.shared_state["last_update"] = time.time()
        
        logger.info(f"✅ Created initial Q&A pair (will be enriched asynchronously)")
        
        # Send initial update to client
        await self._send_update_to_client("qa_pair_created", {
            "question": question,
            "answer": answer,
            "timestamp": qa_timestamp,
            "is_enriched": False,
            "message": f"Q&A pair created - behavioral analysis in progress..."
        })
        
        # Start asynchronous behavioral enrichment (simulates latency)
        asyncio.create_task(self._enrich_qa_pair_async(enriched_qa, qa_timestamp))
    
    async def _enrich_qa_pair_async(self, enriched_qa: EnrichedQAPair, qa_timestamp: float):
        """Asynchronously enrich Q&A pair with behavioral features (simulates latency)"""
        try:
            # Simulate latency (5-10 seconds as mentioned)
            latency = 3.0 + (qa_timestamp % 5.0)  # 3-8 seconds latency
            logger.info(f"⏳ Simulating {latency:.1f}s latency for behavioral analysis...")
            await asyncio.sleep(latency)
            
            # Fetch behavioral features using the configured extractor
            behavioral_features = await self.behavioral_extractor.extract_features_for_timestamp(qa_timestamp)
            
            # Find additional features around this timestamp
            relevant_features = self._find_features_around_timestamp(qa_timestamp)
            
            # Combine all features
            all_features = behavioral_features + relevant_features
            
            # Generate behavioral insights
            behavioral_insights = self._generate_qa_pair_description(
                enriched_qa.question, enriched_qa.answer, all_features
            )
            
            # Update the enriched Q&A pair
            enriched_qa.behavioral_features = all_features
            enriched_qa.behavioral_insights = behavioral_insights
            enriched_qa.confidence = self._calculate_correlation_confidence(all_features)
            enriched_qa.is_enriched = True
            enriched_qa.last_updated = time.time()
            
            # Update shared state
            self.shared_state["enriched_qa_pairs"] = self._get_enriched_qa_pairs_dict()
            self.shared_state["last_qa_pair"] = enriched_qa
            self.shared_state["last_update"] = time.time()
            
            logger.info(f"🎯 Q&A pair enriched with behavioral insights: {behavioral_insights[:100]}...")
            
            # Send enrichment update to client
            await self._send_update_to_client("qa_pair_enriched", {
                "question": enriched_qa.question,
                "answer": enriched_qa.answer,
                "timestamp": qa_timestamp,
                "behavioral_insights": behavioral_insights,
                "confidence": enriched_qa.confidence,
                "features_count": len(all_features),
                "is_enriched": True,
                "message": f"Q&A pair enriched with behavioral analysis"
            })
            
        except Exception as e:
            logger.error(f"❌ Error enriching Q&A pair: {e}")
    
    async def _send_update_to_client(self, update_type: str, data: Dict[str, Any]):
        """Send StateAgent updates to the client via WebSocket"""
        if self.websocket:
            import json
            try:
                message = {
                    "type": "state_agent_update",
                    "update_type": update_type,
                    "data": data,
                    "timestamp": time.time()
                }
                await self.websocket.send(json.dumps(message))
                logger.info(f"✅ Sent StateAgent update to client: {update_type}")
                logger.info(f"📤 Message: {json.dumps(message, indent=2)}")
            except Exception as e:
                logger.error(f"❌ Failed to send StateAgent update to client: {e}")
        else:
            logger.warning("⚠️ No websocket connection available for StateAgent updates")
    
    
    def _generate_qa_pair_description(self, question: str, answer: str, features: List[BehavioralFeature]) -> str:
        """Generate a descriptive contextual analysis for a complete Q&A pair"""
        if not features:
            return f"During the Q&A exchange, no specific behavioral indicators were detected."
        
        # Group features by type
        feature_groups = {}
        for feature in features:
            if feature.feature_type not in feature_groups:
                feature_groups[feature.feature_type] = []
            feature_groups[feature.feature_type].append(feature)
        
        # Build descriptive text
        descriptions = []
        
        for feature_type, type_features in feature_groups.items():
            if feature_type == "facial_expression":
                expressions = [f.value for f in type_features if f.confidence > 0.5]
                if expressions:
                    descriptions.append(f"Facial expressions showed: {', '.join(set(expressions))}")
            
            elif feature_type == "audio_pitch":
                pitches = [f.value for f in type_features if f.confidence > 0.5]
                if pitches:
                    avg_pitch = sum(pitches) / len(pitches)
                    pitch_desc = "elevated" if avg_pitch > 0.6 else "normal" if avg_pitch > 0.4 else "lowered"
                    descriptions.append(f"Voice pitch was {pitch_desc} (avg: {avg_pitch:.2f})")
            
            elif feature_type == "sentiment":
                sentiments = [f.value for f in type_features if f.confidence > 0.5]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_desc = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
                    descriptions.append(f"Overall sentiment was {sentiment_desc} ({avg_sentiment:.2f})")
            
            elif feature_type == "attention":
                attention_levels = [f.value for f in type_features if f.confidence > 0.5]
                if attention_levels:
                    avg_attention = sum(attention_levels) / len(attention_levels)
                    attention_desc = "high" if avg_attention > 0.7 else "moderate" if avg_attention > 0.5 else "low"
                    descriptions.append(f"Attention level was {attention_desc} ({avg_attention:.2f})")
            
            elif feature_type == "stress":
                stress_levels = [f.value for f in type_features if f.confidence > 0.5]
                if stress_levels:
                    avg_stress = sum(stress_levels) / len(stress_levels)
                    stress_desc = "high" if avg_stress > 0.6 else "moderate" if avg_stress > 0.3 else "low"
                    descriptions.append(f"Stress indicators were {stress_desc} ({avg_stress:.2f})")
            
            elif feature_type == "transcription":
                words = [f.value for f in type_features if f.confidence > 0.5]
                if words:
                    descriptions.append(f"Key words detected: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")
        
        if descriptions:
            return f"During the Q&A exchange, behavioral analysis revealed: {'; '.join(descriptions)}."
        else:
            return f"During the Q&A exchange, behavioral features were detected but with low confidence levels."
    
    def _find_features_around_timestamp(self, target_timestamp: float, window_seconds: float = 5.0) -> List[BehavioralFeature]:
        """Find behavioral features within a time window around the target timestamp"""
        relevant_features = []
        
        for feature in self.behavioral_features:
            time_diff = abs(feature.timestamp - target_timestamp)
            if time_diff <= window_seconds:
                relevant_features.append(feature)
        
        # Sort by proximity to target timestamp
        relevant_features.sort(key=lambda f: abs(f.timestamp - target_timestamp))
        
        return relevant_features
    
    def _generate_contextual_description(self, conversation_type: str, content: str, features: List[BehavioralFeature]) -> str:
        """Generate a descriptive contextual analysis of the conversation with behavioral features"""
        if not features:
            return f"During the {conversation_type}, no specific behavioral indicators were detected in the time window."
        
        # Group features by type
        feature_groups = {}
        for feature in features:
            if feature.feature_type not in feature_groups:
                feature_groups[feature.feature_type] = []
            feature_groups[feature.feature_type].append(feature)
        
        # Build descriptive text
        descriptions = []
        
        for feature_type, type_features in feature_groups.items():
            if feature_type == "emotion":
                emotions = [f.value for f in type_features if f.confidence > 0.5]
                if emotions:
                    descriptions.append(f"Emotional state showed: {', '.join(set(emotions))}")
            
            elif feature_type == "sentiment":
                sentiments = [f.value for f in type_features if f.confidence > 0.5]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments) if isinstance(sentiments[0], (int, float)) else sentiments[0]
                    descriptions.append(f"Sentiment analysis indicated: {avg_sentiment}")
            
            elif feature_type == "attention":
                attention_levels = [f.value for f in type_features if f.confidence > 0.5]
                if attention_levels:
                    avg_attention = sum(attention_levels) / len(attention_levels) if isinstance(attention_levels[0], (int, float)) else attention_levels[0]
                    descriptions.append(f"Attention level was: {avg_attention}")
            
            elif feature_type == "stress":
                stress_levels = [f.value for f in type_features if f.confidence > 0.5]
                if stress_levels:
                    avg_stress = sum(stress_levels) / len(stress_levels) if isinstance(stress_levels[0], (int, float)) else stress_levels[0]
                    descriptions.append(f"Stress indicators showed: {avg_stress}")
        
        if descriptions:
            return f"During the {conversation_type} '{content[:50]}...', behavioral analysis revealed: {'; '.join(descriptions)}."
        else:
            return f"During the {conversation_type}, behavioral features were detected but with low confidence levels."
    
    def _calculate_correlation_confidence(self, features: List[BehavioralFeature]) -> float:
        """Calculate confidence in the correlation between conversation and behavioral features"""
        if not features:
            return 0.0
        
        # Average confidence of all features
        total_confidence = sum(f.confidence for f in features)
        return total_confidence / len(features)
    
    def _get_question_insights(self) -> List[Dict[str, Any]]:
        """Get all question-related insights"""
        return [
            {
                "timestamp": insight.conversation_timestamp,
                "content": insight.conversation_content,
                "description": insight.contextual_description,
                "confidence": insight.correlation_confidence,
                "features": [
                    {
                        "type": f.feature_type,
                        "value": f.value,
                        "confidence": f.confidence,
                        "description": f.description
                    } for f in insight.behavioral_features
                ]
            }
            for insight in self.correlated_insights
            if insight.conversation_type == "question"
        ]
    
    def _get_answer_insights(self) -> List[Dict[str, Any]]:
        """Get all answer-related insights"""
        return [
            {
                "timestamp": insight.conversation_timestamp,
                "content": insight.conversation_content,
                "description": insight.contextual_description,
                "confidence": insight.correlation_confidence,
                "features": [
                    {
                        "type": f.feature_type,
                        "value": f.value,
                        "confidence": f.confidence,
                        "description": f.description
                    } for f in insight.behavioral_features
                ]
            }
            for insight in self.correlated_insights
            if insight.conversation_type == "answer"
        ]
    
    def _get_qa_pair_insights(self) -> List[Dict[str, Any]]:
        """Get all Q&A pair-related insights"""
        return [
            {
                "timestamp": insight.conversation_timestamp,
                "content": insight.conversation_content,
                "description": insight.contextual_description,
                "confidence": insight.correlation_confidence,
                "features": [
                    {
                        "type": f.feature_type,
                        "value": f.value,
                        "confidence": f.confidence,
                        "description": f.description
                    } for f in insight.behavioral_features
                ]
            }
            for insight in self.correlated_insights
            if insight.conversation_type == "qa_pair"
        ]
    
    def _get_enriched_qa_pairs_dict(self) -> List[Dict[str, Any]]:
        """Convert enriched Q&A pairs to dictionary format for shared state"""
        return [
            {
                "question": qa.question,
                "answer": qa.answer,
                "timestamp": qa.timestamp,
                "behavioral_insights": qa.behavioral_insights,
                "confidence": qa.confidence,
                "is_enriched": qa.is_enriched,
                "last_updated": qa.last_updated,
                "features_count": len(qa.behavioral_features)
            }
            for qa in self.enriched_qa_pairs
        ]
    
    def add_behavioral_feature(self, feature: BehavioralFeature):
        """Add a new behavioral feature (called by external systems)"""
        self.behavioral_features.append(feature)
        logger.info(f"Added behavioral feature: {feature.feature_type} at {feature.timestamp}")
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of all insights for the conversation"""
        enriched_count = len(self.enriched_qa_pairs)
        enriched_with_insights = len([qa for qa in self.enriched_qa_pairs if qa.is_enriched])
        
        return {
            "total_insights": len(self.correlated_insights),
            "question_insights": len([i for i in self.correlated_insights if i.conversation_type == "question"]),
            "answer_insights": len([i for i in self.correlated_insights if i.conversation_type == "answer"]),
            "qa_pair_insights": len([i for i in self.correlated_insights if i.conversation_type == "qa_pair"]),
            "enriched_qa_pairs": enriched_count,
            "enriched_with_insights": enriched_with_insights,
            "average_confidence": sum(i.correlation_confidence for i in self.correlated_insights) / len(self.correlated_insights) if self.correlated_insights else 0.0,
            "behavioral_features_count": len(self.behavioral_features),
            "last_update": self.shared_state.get("last_update", time.time())
        }
