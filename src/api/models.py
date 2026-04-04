from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Raw message text"
    )
    user_id: str = Field(..., description="UUID of the sending user")
    source: str = Field(default="real", description="Data source: real or synthetic_hf")


class MessageResponse(BaseModel):
    status: str = "accepted"
    message_id: str


class FlagPayload(BaseModel):
    message_id: str = Field(..., description="UUID of the flagged message")
    flagged_by: str | None = Field(default=None, description="UUID of flagging user")
    reason: str | None = Field(default=None, description="Reason for flagging")


class FlagResponse(BaseModel):
    status: str = "accepted"
    flag_id: str
