from dataclasses import dataclass
from datetime import datetime, timedelta

from cryptography.fernet import Fernet
from pydantic import BaseModel


@dataclass
class SymmetricKeyInfo:
    fernet: Fernet
    expiration_time: datetime

    @classmethod
    def create(cls, fernet: Fernet, ttl_seconds: int = 60 * 60 * 5):  # 5 hours
        return cls(fernet, datetime.now() + timedelta(seconds=ttl_seconds))

    def is_expired(self) -> bool:
        return datetime.now() > self.expiration_time


class SymmetricKeyExchange(BaseModel):
    encrypted_symmetric_key: str


class PublicKeyResponse(BaseModel):
    public_key: str
    timestamp: float
