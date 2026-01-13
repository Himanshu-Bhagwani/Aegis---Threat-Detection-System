#!/usr/bin/env python3
"""
Unified Risk Scoring - Fusion Module

Combines threat signals from multiple detection layers:
- GPS Spoofing Detection
- Login Anomaly Detection  
- Password/Hash Integrity
- Transaction Fraud Detection

Outputs a unified risk score per user/event with breakdown.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json


class ThreatCategory(str, Enum):
    """Categories of threats detected by the system."""
    GPS_SPOOFING = "gps_spoofing"
    LOGIN_ANOMALY = "login_anomaly"
    PASSWORD_WEAKNESS = "password_weakness"
    TRANSACTION_FRAUD = "transaction_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    IDENTITY_THEFT = "identity_theft"


class RiskLevel(str, Enum):
    """Risk severity levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreatSignal:
    """Individual threat signal from a detection model."""
    category: ThreatCategory
    probability: float  # 0.0 to 1.0
    confidence: float   # 0.0 to 1.0
    source_model: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UnifiedRiskScore:
    """Combined risk assessment from all detection layers."""
    user_id: Optional[str]
    event_id: Optional[str]
    
    # Overall scores
    unified_score: float           # 0.0 to 1.0
    risk_level: RiskLevel
    confidence: float
    
    # Component scores
    gps_risk: float
    login_risk: float
    password_risk: float
    fraud_risk: float
    
    # Detailed breakdown
    threat_signals: List[Dict]
    primary_threats: List[str]
    recommended_actions: List[str]
    
    # Metadata
    timestamp: str
    models_used: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["risk_level"] = self.risk_level.value
        return result


class RiskFusionEngine:
    """
    Engine for combining multiple threat signals into unified risk scores.
    
    Uses configurable weights and fusion strategies to combine:
    - GPS spoofing probability
    - Login anomaly probability
    - Password weakness/breach probability
    - Transaction fraud probability
    
    Additional signals can be incorporated:
    - Account takeover indicators
    - Identity theft patterns
    - Behavioral anomalies
    """
    
    # Default weights for each signal category
    DEFAULT_WEIGHTS = {
        ThreatCategory.GPS_SPOOFING: 1.5,
        ThreatCategory.LOGIN_ANOMALY: 2.0,
        ThreatCategory.PASSWORD_WEAKNESS: 1.0,
        ThreatCategory.TRANSACTION_FRAUD: 2.5,
        ThreatCategory.ACCOUNT_TAKEOVER: 3.0,
        ThreatCategory.IDENTITY_THEFT: 2.5,
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        RiskLevel.MINIMAL: 0.1,
        RiskLevel.LOW: 0.25,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.HIGH: 0.75,
        RiskLevel.CRITICAL: 0.9,
    }
    
    def __init__(self, weights: Dict[ThreatCategory, float] = None):
        """
        Initialize the fusion engine.
        
        Args:
            weights: Custom weights for each threat category.
                     Higher weights = more influence on final score.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
    def set_weights(self, weights: Dict[ThreatCategory, float]):
        """Update fusion weights."""
        self.weights.update(weights)
    
    def _normalize_probability(self, prob: float) -> float:
        """Ensure probability is in valid range."""
        if prob < 0:
            return 0.0
        return float(np.clip(prob, 0.0, 1.0))
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score."""
        if score >= self.RISK_THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.RISK_THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif score >= self.RISK_THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _weighted_average(self, signals: List[ThreatSignal]) -> float:
        """Compute weighted average of signals."""
        if not signals:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            weight = self.weights.get(signal.category, 1.0)
            # Adjust weight by confidence
            effective_weight = weight * signal.confidence
            weighted_sum += signal.probability * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _max_threat_fusion(self, signals: List[ThreatSignal]) -> float:
        """
        Alternative fusion: take maximum threat probability.
        Useful when any single high threat should trigger alerts.
        """
        if not signals:
            return 0.0
        return max(s.probability for s in signals)
    
    def _bayesian_fusion(self, signals: List[ThreatSignal]) -> float:
        """
        Bayesian fusion of independent threat probabilities.
        P(threat) = 1 - Π(1 - p_i)
        """
        if not signals:
            return 0.0
        
        # Compute probability of NO threat from any source
        p_no_threat = 1.0
        for signal in signals:
            # Weight probabilities by confidence
            effective_prob = signal.probability * signal.confidence
            p_no_threat *= (1.0 - effective_prob)
        
        return 1.0 - p_no_threat
    
    def _generate_recommendations(self, signals: List[ThreatSignal], 
                                   unified_score: float) -> List[str]:
        """Generate actionable recommendations based on threats detected."""
        recommendations = []
        
        # Check each signal for specific recommendations
        for signal in signals:
            if signal.probability < 0.3:
                continue
                
            if signal.category == ThreatCategory.GPS_SPOOFING:
                if signal.probability >= 0.7:
                    recommendations.append("CRITICAL: Potential GPS spoofing detected. Verify user location through alternative means.")
                elif signal.probability >= 0.5:
                    recommendations.append("Request additional location verification (e.g., IP geolocation cross-check).")
            
            elif signal.category == ThreatCategory.LOGIN_ANOMALY:
                if signal.probability >= 0.8:
                    recommendations.append("CRITICAL: Highly anomalous login pattern. Consider immediate session termination.")
                elif signal.probability >= 0.5:
                    recommendations.append("Trigger step-up authentication (MFA/CAPTCHA).")
                    recommendations.append("Review recent account activity for unauthorized access.")
            
            elif signal.category == ThreatCategory.PASSWORD_WEAKNESS:
                if signal.probability >= 0.7:
                    recommendations.append("Password appears in breach databases or is very weak. Force password reset.")
                elif signal.probability >= 0.4:
                    recommendations.append("Recommend user update to a stronger password.")
            
            elif signal.category == ThreatCategory.TRANSACTION_FRAUD:
                if signal.probability >= 0.8:
                    recommendations.append("CRITICAL: High fraud probability. Block transaction and notify user.")
                elif signal.probability >= 0.5:
                    recommendations.append("Flag transaction for manual review before processing.")
                    recommendations.append("Consider requiring additional verification (OTP, biometric).")
        
        # General recommendations based on overall risk
        if unified_score >= 0.9:
            recommendations.insert(0, "🚨 IMMEDIATE ACTION REQUIRED: Multiple high-severity threats detected.")
            recommendations.append("Consider temporary account lock pending investigation.")
        elif unified_score >= 0.7:
            recommendations.append("Increase monitoring frequency for this user/session.")
        
        return recommendations if recommendations else ["No immediate action required. Continue standard monitoring."]
    
    def _identify_primary_threats(self, signals: List[ThreatSignal]) -> List[str]:
        """Identify the primary threat categories."""
        primary = []
        for signal in sorted(signals, key=lambda s: s.probability, reverse=True):
            if signal.probability >= 0.5:
                primary.append(signal.category.value)
        return primary[:3]  # Top 3 threats
    
    def compute_unified_score(
        self,
        gps_score: Optional[Dict] = None,
        login_score: Optional[Dict] = None,
        password_score: Optional[float] = None,
        fraud_score: Optional[float] = None,
        user_id: str = None,
        event_id: str = None,
        fusion_strategy: str = "weighted_average"
    ) -> UnifiedRiskScore:
        """
        Compute unified risk score from all detection modules.
        
        Args:
            gps_score: Output from GPS spoofing detection (dict with spoof_probability, confidence)
            login_score: Output from login anomaly detection (dict with anomaly_probability, confidence)
            password_score: Password weakness/breach probability (float 0-1)
            fraud_score: Transaction fraud probability (float 0-1)
            user_id: Optional user identifier
            event_id: Optional event identifier
            fusion_strategy: "weighted_average", "max_threat", or "bayesian"
            
        Returns:
            UnifiedRiskScore object with complete risk assessment
        """
        signals: List[ThreatSignal] = []
        models_used = []
        
        # Process GPS score
        gps_prob = 0.0
        if gps_score is not None:
            gps_prob = self._normalize_probability(gps_score.get("spoof_probability", 0.0))
            gps_conf = gps_score.get("confidence", 0.7)
            signals.append(ThreatSignal(
                category=ThreatCategory.GPS_SPOOFING,
                probability=gps_prob,
                confidence=gps_conf,
                source_model="gps_ensemble",
                details=gps_score.get("model_scores", {})
            ))
            models_used.extend(gps_score.get("models_used", ["gps"]))
        
        # Process login score
        login_prob = 0.0
        if login_score is not None:
            login_prob = self._normalize_probability(login_score.get("anomaly_probability", 0.0))
            login_conf = login_score.get("confidence", 0.7)
            signals.append(ThreatSignal(
                category=ThreatCategory.LOGIN_ANOMALY,
                probability=login_prob,
                confidence=login_conf,
                source_model="login_ensemble",
                details={
                    "model_scores": login_score.get("model_scores", {}),
                    "rule_scores": login_score.get("rule_scores", {}),
                    "risk_level": login_score.get("risk_level", "unknown")
                }
            ))
            models_used.extend(login_score.get("models_used", ["login"]))
        
        # Process password score
        password_prob = 0.0
        if password_score is not None:
            password_prob = self._normalize_probability(password_score)
            signals.append(ThreatSignal(
                category=ThreatCategory.PASSWORD_WEAKNESS,
                probability=password_prob,
                confidence=0.8,  # ML model confidence
                source_model="password_rf"
            ))
            models_used.append("password_rf")
        
        # Process fraud score
        fraud_prob = 0.0
        if fraud_score is not None:
            fraud_prob = self._normalize_probability(fraud_score)
            signals.append(ThreatSignal(
                category=ThreatCategory.TRANSACTION_FRAUD,
                probability=fraud_prob,
                confidence=0.85,  # XGBoost model confidence
                source_model="fraud_xgb"
            ))
            models_used.append("fraud_xgb")
        
        # Apply fusion strategy
        if fusion_strategy == "max_threat":
            unified = self._max_threat_fusion(signals)
        elif fusion_strategy == "bayesian":
            unified = self._bayesian_fusion(signals)
        else:  # default: weighted_average
            unified = self._weighted_average(signals)
        
        # Calculate overall confidence (average of signal confidences)
        if signals:
            overall_confidence = np.mean([s.confidence for s in signals])
        else:
            overall_confidence = 0.0
        
        # Determine risk level
        risk_level = self._get_risk_level(unified)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(signals, unified)
        
        # Identify primary threats
        primary_threats = self._identify_primary_threats(signals)
        
        return UnifiedRiskScore(
            user_id=user_id,
            event_id=event_id,
            unified_score=float(unified),
            risk_level=risk_level,
            confidence=float(overall_confidence),
            gps_risk=float(gps_prob),
            login_risk=float(login_prob),
            password_risk=float(password_prob),
            fraud_risk=float(fraud_prob),
            threat_signals=[asdict(s) for s in signals],
            primary_threats=primary_threats,
            recommended_actions=recommendations,
            timestamp=datetime.utcnow().isoformat(),
            models_used=list(set(models_used))
        )


# Singleton instance for convenience
_engine = None

def get_fusion_engine() -> RiskFusionEngine:
    """Get or create the global fusion engine instance."""
    global _engine
    if _engine is None:
        _engine = RiskFusionEngine()
    return _engine


def compute_unified_risk(
    gps_score: Optional[Dict] = None,
    login_score: Optional[Dict] = None,
    password_score: Optional[float] = None,
    fraud_score: Optional[float] = None,
    user_id: str = None,
    event_id: str = None,
    fusion_strategy: str = "weighted_average"
) -> Dict:
    """
    Convenience function to compute unified risk score.
    
    Returns dictionary (JSON-serializable) with full risk assessment.
    """
    engine = get_fusion_engine()
    result = engine.compute_unified_score(
        gps_score=gps_score,
        login_score=login_score,
        password_score=password_score,
        fraud_score=fraud_score,
        user_id=user_id,
        event_id=event_id,
        fusion_strategy=fusion_strategy
    )
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    
    # Simulated scores from individual models
    gps_result = {
        "spoof_probability": 0.75,
        "confidence": 0.8,
        "model_scores": {"isolation_forest": 0.7, "gbm": 0.8, "cnn_rnn": 0.75},
        "models_used": ["isolation_forest", "gbm", "cnn_rnn"]
    }
    
    login_result = {
        "anomaly_probability": 0.6,
        "confidence": 0.7,
        "model_scores": {"isolation_forest": 0.5, "gbm": 0.65, "autoencoder": 0.55},
        "rule_scores": {"brute_force_risk": 0.3, "off_hours_risk": 0.4},
        "risk_level": "medium",
        "models_used": ["isolation_forest", "gbm", "autoencoder"]
    }
    
    password_prob = 0.4  # From password scoring
    fraud_prob = 0.85    # From fraud model
    
    # Compute unified risk
    result = compute_unified_risk(
        gps_score=gps_result,
        login_score=login_result,
        password_score=password_prob,
        fraud_score=fraud_prob,
        user_id="user_12345",
        event_id="evt_67890"
    )
    
    print("=" * 60)
    print("UNIFIED RISK ASSESSMENT")
    print("=" * 60)
    print(f"User: {result['user_id']}")
    print(f"Event: {result['event_id']}")
    print(f"\n📊 UNIFIED RISK SCORE: {result['unified_score']:.2%}")
    print(f"🎯 RISK LEVEL: {result['risk_level'].upper()}")
    print(f"🔒 CONFIDENCE: {result['confidence']:.2%}")
    print(f"\n--- Component Scores ---")
    print(f"  GPS Spoofing Risk:    {result['gps_risk']:.2%}")
    print(f"  Login Anomaly Risk:   {result['login_risk']:.2%}")
    print(f"  Password Risk:        {result['password_risk']:.2%}")
    print(f"  Fraud Risk:           {result['fraud_risk']:.2%}")
    print(f"\n⚠️  Primary Threats: {', '.join(result['primary_threats']) or 'None'}")
    print(f"\n📋 RECOMMENDED ACTIONS:")
    for i, action in enumerate(result['recommended_actions'], 1):
        print(f"  {i}. {action}")
    print(f"\n🔧 Models Used: {', '.join(result['models_used'])}")
    print(f"⏰ Timestamp: {result['timestamp']}")
