from regression_model.config.core import config
from regression_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        column_name=config.model_config.CABIN,
    )
    assert sample_input_data["cabin"].iat[0] == "B5"

    # When
    subject = transformer.transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[0] == "B"
