import pytest
from pydantic import TypeAdapter, ValidationError

from deepresearch.domain import HtmlLocator, Locator, PdfLocator


def test_html_locator_uses_half_open_unicode_code_point_offsets() -> None:
    text = "a多模态z"
    locator = HtmlLocator(paragraph_id="p-1", start_char=1, end_char=4)

    assert text[locator.start_char : locator.end_char] == "多模态"


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (HtmlLocator, {"paragraph_id": "", "start_char": 0, "end_char": 1}),
        (HtmlLocator, {"paragraph_id": "p", "start_char": 1, "end_char": 1}),
        (
            PdfLocator,
            {"page_index": -1, "block_index": 0, "start_char": 0, "end_char": 1},
        ),
        (
            PdfLocator,
            {"page_index": 0, "block_index": 0, "start_char": 2, "end_char": 1},
        ),
    ],
)
def test_locators_reject_invalid_ranges_and_coordinates(
    model: type[HtmlLocator] | type[PdfLocator], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_locator_union_uses_kind_discriminator() -> None:
    adapter = TypeAdapter(Locator)

    html = adapter.validate_python(
        {"kind": "html", "paragraph_id": "p", "start_char": 0, "end_char": 1}
    )
    pdf = adapter.validate_python(
        {
            "kind": "pdf",
            "page_index": 0,
            "block_index": 1,
            "start_char": 0,
            "end_char": 2,
        }
    )

    assert isinstance(html, HtmlLocator)
    assert isinstance(pdf, PdfLocator)


def test_locator_models_are_frozen_and_forbid_extra_fields() -> None:
    locator = HtmlLocator(paragraph_id="p", start_char=0, end_char=1)

    with pytest.raises(ValidationError):
        HtmlLocator(paragraph_id="p", start_char=0, end_char=1, section="intro")
    with pytest.raises(ValidationError):
        locator.end_char = 2
