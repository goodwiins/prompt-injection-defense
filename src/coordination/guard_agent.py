from typing import Dict, Any, Optional, List
import time
import structlog
from src.detection.embedding_classifier import EmbeddingClassifier
from src.detection.ensemble_classifier import EnsembleClassifier
from src.detection.patterns import PatternDetector
from src.response.circuit_breaker import CircuitBreaker, AlertSeverity
from src.coordination.quarantine import QuarantineManager
from src.coordination.policy_enforcer import PolicyEnforcer
from src.coordination.behavioral_monitor import BehavioralMonitor
from src.coordination.behavioral_monitor import BehavioralMonitor
from src.coordination.peerguard import PeerGuard
from src.coordination.ovon_protocol import OVONMessage

logger = structlog.get_logger()

class GuardAgent:
    """
    Advanced coordination agent that orchestrates the complete defense pipeline
    with integrated response coordination and multi-agent capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, agent_id: str = "guard_agent_001"):
        self.config = config or {}
        self.agent_id = agent_id

        # Initialize detection components
        self.pattern_detector = PatternDetector()

        # Check if ensemble is enabled in config
        use_ensemble = self.config.get("detection", {}).get("use_ensemble", False)

        if use_ensemble:
            # Use ensemble classifier for production-grade detection
            model_config = self.config.get("detection", {})
            self.classifier = EnsembleClassifier(
                fast_model_name=model_config.get("fast_model", "all-MiniLM-L6-v2"),
                deep_model_name=model_config.get("deep_model", "all-mpnet-base-v2"),
                specialized_model_name=model_config.get("specialized_model"),
                fast_threshold=model_config.get("fast_threshold", 0.5),
                deep_threshold=model_config.get("threshold", 0.85),
                use_cascade=model_config.get("use_cascade", True),
                model_dir=self.config.get("model_dir", "models")
            )
        else:
            # Use single embedding classifier for compatibility
            model_name = self.config.get("detection", {}).get("fast_model", "all-MiniLM-L6-v2")
            threshold = self.config.get("detection", {}).get("threshold", 0.85)
            model_path = self.config.get("detection", {}).get("model_path")

            self.classifier = EmbeddingClassifier(
                model_name=model_name,
                threshold=threshold,
                model_dir=self.config.get("model_dir", "models")
            )

        # Initialize response coordination components
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.get("circuit_breaker", {}).get("failure_threshold", 5),
            time_window=self.config.get("circuit_breaker", {}).get("recovery_timeout", 60)
        )

        self.quarantine_manager = QuarantineManager(
            default_timeout=self.config.get("quarantine", {}).get("default_duration", 300)
        )

        self.policy_enforcer = PolicyEnforcer()
        self.behavioral_monitor = BehavioralMonitor()

        # Initialize multi-agent coordination components
        self.peerguard = PeerGuard(
            consistency_threshold=self.config.get("peerguard", {}).get("confidence_threshold", 0.8)
        )

        # Load models if specified
        model_path = self.config.get("detection", {}).get("model_path")
        if model_path and not use_ensemble:
            try:
                self.classifier.load_model(model_path)
                logger.info("Loaded trained model", path=model_path)
            except Exception as e:
                logger.error("Failed to load model", path=model_path, error=str(e))

        # Statistics tracking
        self.stats = {
            "total_analyses": 0,
            "blocked_requests": 0,
            "quarantined_agents": 0,
            "circuit_breaker_trips": 0,
            "peer_validations": 0
        }

        logger.info("GuardAgent initialized",
                   agent_id=agent_id,
                   use_ensemble=use_ensemble,
                   response_coordination=True)

    def process_message(self, message: OVONMessage) -> Dict[str, Any]:
        """
        Process an incoming OVON message with full security verification.
        
        Args:
            message: The OVON message to process
            
        Returns:
            Analysis result and decision
        """
        # 1. Verify Protocol Safety (Signature, Trust Level)
        if not message.is_safe():
            logger.warning("Message rejected by protocol safety check", 
                          message_id=message.message_id,
                          source=message.source_agent)
            return self._create_block_response("Protocol violation: Invalid signature or trust level")

        # 2. Check Quarantine
        if self.quarantine_manager.is_quarantined(message.source_agent):
             logger.warning("Message rejected from quarantined agent", 
                          agent_id=message.source_agent)
             return self._create_block_response(f"Source agent {message.source_agent} is quarantined")
        
        # 3. Analyze Content
        # Pass message metadata into context
        context = {
            "message_id": message.message_id, 
            "source_agent": message.source_agent,
            "llm_tag": message.llm_tag.dict() if message.llm_tag else None
        }
        
        return self.analyze(message.content.utterance, context=context)

    def _create_block_response(self, reason: str) -> Dict[str, Any]:
        """Helper to create a standard block response."""
        return {
            "agent_id": self.agent_id,
            "is_safe": False,
            "confidence": 1.0,
            "recommendation": "block",
            "response_actions": {
                "actions_taken": ["blocked_pre_analysis"],
                "reason": reason
            },
            "system_status": {
                "circuit_breaker_open": self.circuit_breaker.is_open(),
                "quarantined_agents": len(self.quarantine_manager.quarantined_agents),
                "stats": self.stats.copy()
            }
        }

    def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive prompt analysis with integrated response coordination.

        Args:
            prompt: The input text to analyze
            context: Optional context information (user_id, session_id, etc.)

        Returns:
            Dictionary with comprehensive analysis results and response actions
        """
        start_time = time.time()
        self.stats["total_analyses"] += 1

        logger.info("Starting comprehensive analysis",
                   agent_id=self.agent_id,
                   prompt_length=len(prompt))

        # Check circuit breaker status first
        if self.circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, blocking request")
            return self._create_circuit_breaker_response()

        # Phase 1: Pattern Detection (Fastest)
        pattern_result = self.pattern_detector.detect(prompt)

        # Phase 2: Advanced Classification
        if isinstance(self.classifier, EnsembleClassifier):
            # Use ensemble for production-grade detection
            ensemble_results = self.classifier.predict([prompt])
            embedding_result = ensemble_results[0]
            embedding_score = embedding_result["score"]
            embedding_is_injection = embedding_result["is_injection"]
            detection_path = embedding_result.get("detection_path", "fast")
            confidence = embedding_result.get("confidence", 0.0)
        else:
            # Use single classifier
            embedding_probs = self.classifier.predict_proba([prompt])[0]
            embedding_score = float(embedding_probs[1])
            embedding_is_injection = embedding_score >= self.classifier.threshold
            detection_path = "single"
            confidence = max(embedding_score, 1 - embedding_score)

        # Phase 3: Policy Enforcement
        message_data = {
            "content": prompt,
            "context": context,
            "embedding_score": embedding_score,
            "is_injection": embedding_is_injection
        }
        policy_result = self.policy_enforcer.enforce(message_data)
        policy_violation = policy_result.get("violations", [])
        policy_severity = float(policy_result.get("severity_level", 0)) / 4.0

        # Phase 4: Behavioral Analysis
        # For now, use simple behavioral heuristics
        behavior_anomaly = False
        behavior_score = 0.0

        # Check for repeated patterns in context
        if context and "session_id" in context:
            session_id = context["session_id"]
            # This would normally use behavioral_monitor to track patterns
            # For now, skip complex behavioral analysis

        behavior_result = {
            "is_anomaly": behavior_anomaly,
            "score": behavior_score
        }

        # Phase 5: Multi-Agent Peer Validation (if enabled)
        peer_validation = None
        if self.config.get("multi_agent", {}).get("enable_peerguard", False):
            peer_validation = self.peerguard.validate_request(
                prompt=prompt,
                local_result={
                    "is_injection": embedding_is_injection,
                    "confidence": confidence,
                    "pattern_result": pattern_result
                },
                context=context
            )
            self.stats["peer_validations"] += 1

        # Phase 6: Integrated Decision Making
        is_safe, confidence, recommendation = self._make_integrated_decision(
            pattern_result, embedding_is_injection, embedding_score,
            policy_violation, policy_severity,
            behavior_anomaly, behavior_score,
            peer_validation
        )

        # Phase 7: Response Coordination
        response_actions = self._coordinate_response(
            is_safe, confidence, recommendation, context
        )

        # Update behavioral monitor with result (skip for now)
        # self.behavioral_monitor.record_outcome(prompt, is_safe, context)

        # Compile comprehensive result
        analysis_time = (time.time() - start_time) * 1000

        result = {
            "agent_id": self.agent_id,
            "is_safe": is_safe,
            "confidence": confidence,
            "recommendation": recommendation,
            "detection_path": detection_path,
            "response_actions": response_actions,
            "analysis_time_ms": analysis_time,
            "details": {
                "pattern_analysis": pattern_result,
                "embedding_analysis": {
                    "score": embedding_score,
                    "threshold": getattr(self.classifier, 'threshold', 0.85),
                    "detection_path": detection_path
                },
                "policy_analysis": policy_result,
                "behavioral_analysis": behavior_result,
                "peer_validation": peer_validation
            },
            "system_status": {
                "circuit_breaker_open": self.circuit_breaker.is_open(),
                "quarantined_agents": len(self.quarantine_manager.quarantined_agents),
                "stats": self.stats.copy()
            }
        }

        logger.info("Comprehensive analysis complete",
                   agent_id=self.agent_id,
                   is_safe=is_safe,
                   confidence=confidence,
                   recommendation=recommendation,
                   analysis_time_ms=analysis_time)

        return result

    def _make_integrated_decision(self, pattern_result: Dict, embedding_is_injection: bool,
                                 embedding_score: float, policy_violation: bool,
                                 policy_severity: float, behavior_anomaly: bool,
                                 behavior_score: float, peer_validation: Optional[Dict]) -> tuple:
        """
        Make integrated security decision using all available information.
        """
        # Base decision from pattern and embedding analysis
        pattern_suspicious = pattern_result["is_suspicious"]
        pattern_severity = pattern_result["severity"]

        # Combine detection scores
        detection_scores = []
        if pattern_suspicious:
            detection_scores.append(pattern_severity)
        if embedding_is_injection:
            detection_scores.append(embedding_score)

        # Incorporate policy violations
        if policy_violation:
            detection_scores.append(policy_severity)

        # Incorporate behavioral anomalies
        if behavior_anomaly:
            detection_scores.append(behavior_score * 0.8)  # Weight behavioral signals lower

        # Incorporate peer validation
        if peer_validation:
            peer_confidence = peer_validation.get("confidence", 0.0)
            peer_is_threat = peer_validation.get("is_threat", False)
            if peer_is_threat:
                detection_scores.append(peer_confidence * 0.9)  # High weight for peer validation

        # Calculate overall confidence
        if detection_scores:
            # Use weighted average, giving more weight to higher scores
            weights = [score ** 2 for score in detection_scores]
            confidence = sum(s * w for s, w in zip(detection_scores, weights)) / sum(weights)
        else:
            confidence = 0.0

        # Make safety decision
        high_confidence_threshold = self.config.get("decision", {}).get("high_confidence_threshold", 0.8)
        low_confidence_threshold = self.config.get("decision", {}).get("low_confidence_threshold", 0.3)

        is_safe = confidence < low_confidence_threshold
        recommendation = "allow"

        if not is_safe:
            if confidence >= high_confidence_threshold:
                recommendation = "block"
            else:
                recommendation = "flag_for_review"

            # Escalate to block for policy violations
            if policy_violation and policy_severity > 0.7:
                recommendation = "block"

            # Escalate for strong peer validation
            if peer_validation and peer_validation.get("is_threat") and peer_validation.get("confidence", 0) > 0.9:
                recommendation = "block"

        return is_safe, confidence, recommendation

    def _coordinate_response(self, is_safe: bool, confidence: float,
                           recommendation: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Coordinate response actions based on the security decision.
        """
        actions = []

        if not is_safe:
            self.stats["blocked_requests"] += 1

            # Circuit breaker coordination
            if confidence > 0.9:
                severity = AlertSeverity.CRITICAL if confidence > 0.95 else AlertSeverity.HIGH
                self.circuit_breaker.record_alert(
                    severity=severity,
                    source=self.agent_id,
                    category="security_threat",
                    details={
                        "confidence": confidence,
                        "recommendation": recommendation
                    }
                )
                if self.circuit_breaker.is_open():
                    self.stats["circuit_breaker_trips"] += 1
                    actions.append("circuit_breaker_tripped")

            # Quarantine coordination for severe threats
            if confidence > 0.95 and context:
                user_id = context.get("user_id")
                session_id = context.get("session_id")
                if user_id or session_id:
                    self.quarantine_manager.isolate(
                        agent_id=user_id or session_id,
                        duration=600  # 10 minutes
                    )
                    quarantine_id = user_id or session_id
                    self.stats["quarantined_agents"] += 1
                    actions.append(f"quarantined:{quarantine_id}")

            # Policy enforcement actions
            policy_actions = self.policy_enforcer.get_response_actions(recommendation, confidence)
            actions.extend(policy_actions)

        return {
            "actions_taken": actions,
            "circuit_breaker_status": "open" if self.circuit_breaker.is_open() else "closed",
            "quarantine_active": len(self.quarantine_manager.quarantined_agents) > 0
        }

    def _create_circuit_breaker_response(self) -> Dict[str, Any]:
        """Create response when circuit breaker is open."""
        return {
            "agent_id": self.agent_id,
            "is_safe": False,
            "confidence": 1.0,
            "recommendation": "block",
            "response_actions": {
                "actions_taken": ["circuit_breaker_protection"],
                "circuit_breaker_status": "open",
                "quarantine_active": False
            },
            "details": {
                "reason": "Circuit breaker is open due to repeated threats",
                "recovery_time": self.circuit_breaker.get_recovery_time()
            },
            "system_status": {
                "circuit_breaker_open": True,
                "stats": self.stats.copy()
            }
        }
