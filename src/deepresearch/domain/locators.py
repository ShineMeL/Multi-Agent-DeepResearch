from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self, TypeAlias, override

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        values.update(update)
        return type(self).model_validate(values)


class HtmlLocator(_DomainModel):
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


class PdfLocator(_DomainModel):
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
