from src.base.schema import Fields, Schema


content_not_required = Schema("Tests", required=False).append(Fields.Doc.content.set_str_value(required=False))

# Define Doc model
Doc = Schema("Doc").append(
    Fields.Doc.docid.set_str_value(
        min_length=1,
        regex="^docid",
    ),
    Fields.Doc.content,
    Fields.Doc.headline,
    Fields.Senti.score.set_num_value(
        required=False,
        gt=0,
        lt=1,
    ),
    content_not_required.to_list("list_of_tests", required=False),  # List of content_not_required, key name is list_of_tests
)

# Define information
Doc.content.set_description("Text to be processed; could be in Chinese or in English")

# Define Doc examples
doc_example_1 = Doc.new_example(
    docid="docid001",  # Field example
    headline="Example headline!",  # Field example
    content="Example content.",  # Field example
    summary="Example summary #1.",  # Example's summary
    description='description": "# Example **description** #1.',  # Example's description
)

doc_example_2 = Doc.new_example(docid="docid002", headline="Example headline!", content="Example content.")
examples = {"1": doc_example_1, "2": doc_example_2}
