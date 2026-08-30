from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HtmlLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["html"] = "html"
    paragraph_id: Annotated[str, Field(min_length=1)]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class PdfLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["pdf"] = "pdf"
    page_index: Annotated[int, Field(ge=0)]
    block_index: Annotated[int, Field(ge=0)]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


Locator: TypeAlias = Annotated[  # noqa: UP040 - exact frozen public contract
    HtmlLocator | PdfLocator, Field(discriminator="kind")
]

__all__ = ["HtmlLocator", "Locator", "PdfLocator"]
