import uuid

from fastapi import APIRouter

from ..models import FlagPayload, FlagResponse

router = APIRouter()


@router.post("/flags", response_model=FlagResponse)
async def create_flag(payload: FlagPayload):
    return FlagResponse(status="accepted", flag_id=str(uuid.uuid4()))
