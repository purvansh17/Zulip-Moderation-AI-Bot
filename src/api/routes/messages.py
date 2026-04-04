import uuid

from fastapi import APIRouter

from ..models import MessagePayload, MessageResponse

router = APIRouter()


@router.post("/messages", response_model=MessageResponse)
async def create_message(payload: MessagePayload):
    return MessageResponse(status="accepted", message_id=str(uuid.uuid4()))
