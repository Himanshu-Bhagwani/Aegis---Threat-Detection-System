"""
Password Risk Assessment API Router

Endpoints for evaluating password strength and breach risk.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, SecretStr
from typing import List, Dict, Optional

router = APIRouter()


class PasswordScoreRequest(BaseModel):
    """Request body for password scoring."""
    password: str = Field(..., min_length=1, description="Password to evaluate")
    check_breach: bool = Field(True, description="Whether to check against breach databases")
    
    class Config:
        json_schema_extra = {
            "example": {
                "password": "MyP@ssw0rd123!",
                "check_breach": True
            }
        }


class PasswordHashRequest(BaseModel):
    """Request body for hash-based password checking."""
    password_hash: str = Field(..., description="SHA-1 hash of the password (first 5 chars for k-anonymity)")
    hash_type: str = Field("sha1_prefix", description="Type of hash: sha1_prefix, sha1_full")


class PasswordScoreResponse(BaseModel):
    """Response from password risk assessment."""
    breach_probability: float = Field(..., ge=0, le=1, description="Probability password is in breach databases")
    strength_score: float = Field(..., ge=0, le=1, description="Password strength score (1 = strong)")
    risk_level: str = Field(..., description="Risk level: minimal, low, medium, high, critical")
    features: Dict[str, float] = Field(..., description="Password feature breakdown")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    entropy_bits: float = Field(..., description="Estimated entropy in bits")


class BatchPasswordRequest(BaseModel):
    """Request body for batch password scoring."""
    passwords: List[str] = Field(..., min_length=1, max_length=100)


# Lazy import
_score_password = None
_password_features = None

def get_score_function():
    global _score_password, _password_features
    if _score_password is None:
        from src.passwords.score_password import score_password, password_to_features, estimate_entropy
        _score_password = score_password
        _password_features = {
            "to_features": password_to_features,
            "entropy": estimate_entropy
        }
    return _score_password, _password_features


def get_risk_level(prob: float) -> str:
    """Convert probability to risk level."""
    if prob >= 0.8:
        return "critical"
    elif prob >= 0.6:
        return "high"
    elif prob >= 0.4:
        return "medium"
    elif prob >= 0.2:
        return "low"
    else:
        return "minimal"


def generate_recommendations(features: Dict, prob: float) -> List[str]:
    """Generate password improvement recommendations."""
    recommendations = []
    
    if features.get("length", 0) < 12:
        recommendations.append("Increase password length to at least 12 characters")
    
    if features.get("has_upper", 0) == 0:
        recommendations.append("Add uppercase letters")
    
    if features.get("has_lower", 0) == 0:
        recommendations.append("Add lowercase letters")
    
    if features.get("has_digit", 0) == 0:
        recommendations.append("Include numbers")
    
    if features.get("has_symbol", 0) == 0:
        recommendations.append("Add special characters (!@#$%^&*)")
    
    if features.get("has_seq_pattern", 0) == 1:
        recommendations.append("Avoid sequential patterns (1234, abcd, qwerty)")
    
    if features.get("repeat_ratio", 0) > 0.3:
        recommendations.append("Reduce repeated characters")
    
    if features.get("entropy_est", 0) < 40:
        recommendations.append("Use a more random combination of characters")
    
    if prob >= 0.5:
        recommendations.insert(0, "⚠️ This password may appear in breach databases. Choose a unique password.")
    
    if not recommendations:
        recommendations.append("✓ Password meets basic security requirements")
    
    return recommendations


@router.post("/score", response_model=PasswordScoreResponse, summary="Score password strength and breach risk")
async def score_password(request: PasswordScoreRequest):
    """
    Evaluate a password for strength and breach risk.
    
    The endpoint uses an ML model trained to identify weak/breached passwords
    based on features like:
    - Length and character diversity
    - Presence of sequential patterns
    - Character type distribution
    - Entropy estimation
    
    Returns a breach probability score [0-1] where higher values indicate
    higher likelihood the password is weak or has been breached.
    
    **Privacy Note**: Passwords are not stored or logged. For additional
    privacy, use the /score/hash endpoint with k-anonymity.
    """
    try:
        score_fn, features_fn = get_score_function()
        
        password = request.password
        
        # Get breach probability from ML model
        breach_prob = score_fn(password)
        
        # Get feature breakdown
        features = features_fn["to_features"](password)
        entropy = features_fn["entropy"](password)
        
        # Calculate strength score (inverse of breach probability, adjusted)
        strength = 1.0 - (breach_prob * 0.7)  # Breach accounts for 70%
        if features["length"] >= 16:
            strength = min(strength + 0.1, 1.0)
        if features["unique_chars"] >= 10:
            strength = min(strength + 0.05, 1.0)
        
        # Get risk level
        risk_level = get_risk_level(breach_prob)
        
        # Generate recommendations
        recommendations = generate_recommendations(features, breach_prob)
        
        return PasswordScoreResponse(
            breach_probability=breach_prob,
            strength_score=strength,
            risk_level=risk_level,
            features=features,
            recommendations=recommendations,
            entropy_bits=entropy
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Password scoring model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/score/batch", summary="Score multiple passwords")
async def score_passwords_batch(request: BatchPasswordRequest):
    """
    Score multiple passwords in a single request.
    
    Returns individual scores and aggregate statistics.
    """
    try:
        score_fn, features_fn = get_score_function()
        
        results = []
        for password in request.passwords:
            prob = score_fn(password)
            features = features_fn["to_features"](password)
            results.append({
                "breach_probability": prob,
                "risk_level": get_risk_level(prob),
                "length": features["length"],
                "entropy": features["entropy_est"]
            })
        
        # Aggregate stats
        probs = [r["breach_probability"] for r in results]
        
        return {
            "results": results,
            "total_passwords": len(results),
            "high_risk_count": sum(1 for r in results if r["risk_level"] in ["high", "critical"]),
            "avg_breach_probability": sum(probs) / len(probs),
            "max_breach_probability": max(probs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")


@router.post("/analyze", summary="Get detailed password analysis without ML scoring")
async def analyze_password(request: PasswordScoreRequest):
    """
    Get detailed password feature analysis without ML model scoring.
    
    Useful for quick strength assessment without model inference.
    """
    try:
        _, features_fn = get_score_function()
        
        password = request.password
        features = features_fn["to_features"](password)
        entropy = features_fn["entropy"](password)
        
        # Rule-based strength calculation
        score = 0.0
        
        # Length scoring (0-0.4)
        if features["length"] >= 16:
            score += 0.4
        elif features["length"] >= 12:
            score += 0.3
        elif features["length"] >= 8:
            score += 0.2
        else:
            score += 0.1
        
        # Character diversity (0-0.3)
        diversity = features["has_lower"] + features["has_upper"] + features["has_digit"] + features["has_symbol"]
        score += diversity * 0.075
        
        # Unique characters (0-0.15)
        unique_ratio = features["unique_chars"] / max(features["length"], 1)
        score += min(unique_ratio * 0.15, 0.15)
        
        # Entropy bonus (0-0.15)
        if entropy >= 60:
            score += 0.15
        elif entropy >= 40:
            score += 0.1
        elif entropy >= 20:
            score += 0.05
        
        # Penalties
        if features["has_seq_pattern"]:
            score -= 0.15
        if features["repeat_ratio"] > 0.3:
            score -= 0.1
        
        score = max(0.0, min(1.0, score))
        
        return {
            "strength_score": score,
            "features": features,
            "entropy_bits": entropy,
            "character_diversity": diversity,
            "unique_ratio": unique_ratio,
            "issues": {
                "too_short": features["length"] < 8,
                "no_uppercase": features["has_upper"] == 0,
                "no_lowercase": features["has_lower"] == 0,
                "no_numbers": features["has_digit"] == 0,
                "no_symbols": features["has_symbol"] == 0,
                "sequential_pattern": features["has_seq_pattern"] == 1,
                "high_repetition": features["repeat_ratio"] > 0.3
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/health", summary="Check password scoring service health")
async def health_check():
    """Check if password scoring model is loaded and ready."""
    try:
        score_fn, _ = get_score_function()
        
        # Test with sample password
        test_score = score_fn("TestPassword123!")
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_score": test_score
        }
        
    except FileNotFoundError:
        return {
            "status": "degraded",
            "model_loaded": False,
            "message": "ML model not found. Rule-based scoring still available."
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


