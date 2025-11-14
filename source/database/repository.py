"""
Repository Pattern for Database Operations.

Provides abstraction layer over database access with:
- CRUD operations
- Query optimization
- Transaction management
- Business logic separation
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from .models import User, PredictionLog, ModelMetadata, APIKeyModel, AuditLog


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class

    def create(self, **kwargs) -> Any:
        """Create new record."""
        obj = self.model_class(**kwargs)
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj

    def get_by_id(self, id: Any) -> Optional[Any]:
        """Get record by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get all records with pagination."""
        return self.session.query(self.model_class).offset(skip).limit(limit).all()

    def update(self, id: Any, **kwargs) -> Optional[Any]:
        """Update record."""
        obj = self.get_by_id(id)
        if obj:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            self.session.commit()
            self.session.refresh(obj)
        return obj

    def delete(self, id: Any) -> bool:
        """Delete record."""
        obj = self.get_by_id(id)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False


class UserRepository(BaseRepository):
    """Repository for User operations."""

    def __init__(self, session: Session):
        super().__init__(session, User)

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.session.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(User.email == email).first()

    def get_active_users(self) -> List[User]:
        """Get all active users."""
        return self.session.query(User).filter(User.is_active == True).all()

    def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.session.commit()


class PredictionRepository(BaseRepository):
    """Repository for Prediction operations."""

    def __init__(self, session: Session):
        super().__init__(session, PredictionLog)

    def get_by_model(self, model_id: str, limit: int = 100) -> List[PredictionLog]:
        """Get predictions for specific model."""
        return self.session.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).order_by(desc(PredictionLog.created_at)).limit(limit).all()

    def get_by_user(self, user_id: str, limit: int = 100) -> List[PredictionLog]:
        """Get predictions by user."""
        return self.session.query(PredictionLog).filter(
            PredictionLog.user_id == user_id
        ).order_by(desc(PredictionLog.created_at)).limit(limit).all()

    def get_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get prediction statistics for model."""
        from sqlalchemy import func

        stats = self.session.query(
            func.count(PredictionLog.id).label("total_predictions"),
            func.avg(PredictionLog.inference_time_ms).label("avg_inference_time"),
            func.min(PredictionLog.inference_time_ms).label("min_inference_time"),
            func.max(PredictionLog.inference_time_ms).label("max_inference_time"),
        ).filter(PredictionLog.model_id == model_id).first()

        return {
            "total_predictions": stats.total_predictions or 0,
            "avg_inference_time": float(stats.avg_inference_time or 0),
            "min_inference_time": float(stats.min_inference_time or 0),
            "max_inference_time": float(stats.max_inference_time or 0),
        }


class ModelRepository(BaseRepository):
    """Repository for Model operations."""

    def __init__(self, session: Session):
        super().__init__(session, ModelMetadata)

    def get_by_model_id(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model by model_id."""
        return self.session.query(ModelMetadata).filter(
            ModelMetadata.model_id == model_id
        ).first()

    def get_by_stage(self, stage: str) -> List[ModelMetadata]:
        """Get models by stage."""
        return self.session.query(ModelMetadata).filter(
            and_(
                ModelMetadata.stage == stage,
                ModelMetadata.is_active == True
            )
        ).order_by(desc(ModelMetadata.created_at)).all()

    def get_production_models(self) -> List[ModelMetadata]:
        """Get all production models."""
        return self.get_by_stage("production")

    def promote_model(self, model_id: str, new_stage: str) -> Optional[ModelMetadata]:
        """Promote model to new stage."""
        model = self.get_by_model_id(model_id)
        if model:
            model.stage = new_stage
            model.updated_at = datetime.utcnow()
            if new_stage == "production":
                model.deployed_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(model)
        return model
