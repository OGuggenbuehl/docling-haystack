import json
from unittest.mock import MagicMock

from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument
from haystack.dataclasses.byte_stream import ByteStream

from docling_haystack.converter import DoclingConverter, ExportType

PATHS = ["foo.pdf"]

DL_DOC_JSON = "test/data/2408.09869v5.json"
MD_EXPORT = "test/data/2408.09869v5.md"


# TODO should not have to provide specific conversion mock res (chunking should suffice)
def test_convert_doc_chunks(monkeypatch):
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    INPUT_FILE = DL_DOC_JSON
    EXPECTED_OUT_FILE = "test/data/2408.09869v5_hs_docs_doc_chunks.json"
    with open(INPUT_FILE) as f:
        data_json = f.read()
    mock_dl_doc = DoclingDocument.model_validate_json(data_json)
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert",
        lambda *args, **kwargs: mock_response,
    )

    # TODO: chunking is currently performed; it should best be mocked instead
    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = converter.run(paths=PATHS)["documents"]
    act_data = dict(root=[d.to_dict() for d in docs])
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


# TODO should not have to provide specific conversion mock res (export should suffice)
def test_convert_markdown(monkeypatch):
    INPUT_FILE = MD_EXPORT
    EXPECTED_OUT_FILE = "test/data/2408.09869v5_hs_docs_markdown.json"
    with open(INPUT_FILE) as f:
        mock_md_data = f.read()

    with open("test/data/2408.09869v5.json") as f:
        data_json = f.read()
    mock_dl_doc = DoclingDocument.model_validate_json(data_json)
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert",
        lambda *args, **kwargs: mock_response,
    )

    monkeypatch.setattr(
        "docling_core.types.doc.document.DoclingDocument.export_to_markdown",
        lambda *args, **kwargs: mock_md_data,
    )

    # TODO: chunking is currently performed; it should best be mocked instead
    converter = DoclingConverter(
        export_type=ExportType.MARKDOWN,
    )
    docs = converter.run(paths=PATHS)["documents"]
    act_data = dict(root=[d.to_dict() for d in docs])
    with open(EXPECTED_OUT_FILE) as f:
        exp_data = json.load(fp=f)
    assert exp_data == act_data


def test_serialization_deserialization():
    """Test component serialization and deserialization."""
    converter = DoclingConverter(
        convert_kwargs={"optimize_ocr": True},
        md_export_kwargs={"image_placeholder": "[IMAGE]"},
    )

    # serialize the component to dict
    serialized = converter.to_dict()

    assert "init_parameters" in serialized
    assert serialized["init_parameters"].get("convert_kwargs") == {"optimize_ocr": True}

    md_export_kwargs = serialized["init_parameters"].get("md_export_kwargs", {})
    assert md_export_kwargs.get("image_placeholder") == "[IMAGE]"

    # deserialize back to component
    deserialized = DoclingConverter.from_dict(serialized)
    assert deserialized._convert_kwargs == {"optimize_ocr": True}

    assert deserialized._md_export_kwargs.get("image_placeholder") == "[IMAGE]"


def test_bytestream_handling(monkeypatch):
    """Test conversion from ByteStream."""
    with open("test/data/2408.09869v5.md", "rb") as f:
        data = f.read()

    bytestream = ByteStream(
        data=data,
        meta={"file_extension": "md", "filename": "test_file.md"},
    )

    with open("test/data/2408.09869v5.json") as f:
        data_json = f.read()
    mock_dl_doc = DoclingDocument.model_validate_json(data_json)
    mock_response = MagicMock()
    mock_response.document = mock_dl_doc

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.__init__",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter.convert",
        lambda *args, **kwargs: mock_response,
    )

    def mock_extract_meta(self, dl_doc):
        return {"custom_field": "test_value"}

    monkeypatch.setattr(
        "docling_haystack.converter.MetaExtractor.extract_dl_doc_meta",
        mock_extract_meta,
    )

    converter = DoclingConverter(
        export_type=ExportType.MARKDOWN,
    )

    result = converter.run(paths=[bytestream])
    documents = result["documents"]

    assert len(documents) > 0
    assert documents[0].meta.get("custom_field") == "test_value"
    assert len(documents[0].content) > 0
