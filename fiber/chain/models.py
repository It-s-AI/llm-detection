from enum import Enum
from typing import TypeAlias, TypedDict

from pydantic import BaseModel


class Node(BaseModel):
    hotkey: str
    coldkey: str
    node_id: int
    incentive: float
    netuid: int
    stake: float
    trust: float
    vtrust: float
    last_updated: float
    ip: str
    ip_type: int
    port: int
    protocol: int = 4


class ParamWithTypes(TypedDict):
    name: str
    type: str


class CommitmentDataFieldType(Enum):
    RAW = "Raw"
    BLAKE_TWO_256 = "BlakeTwo256"
    SHA_256 = "Sha256"
    KECCAK_256 = "Keccak256"
    SHA_THREE_256 = "ShaThree256"


CommitmentDataField: TypeAlias = tuple[CommitmentDataFieldType, bytes] | None


class CommitmentQuery(BaseModel):
    fields: list[CommitmentDataField]
    block: int
    deposit: int


class RawCommitmentQuery(BaseModel):
    data: bytes
    block: int
    deposit: int
