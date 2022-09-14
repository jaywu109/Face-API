from src.base.schema import Fields, Schema


# Define senti result
SentiResult = Schema("SentiResult").append(
    Fields.Senti.score,
    Fields.Senti.label,
    Schema("scores").append(
        Fields.Senti.positive_score.set_description("xxx").set_num_value(default=1.0),  # Define information at fields level
        Fields.Senti.neutral_score,
        Fields.Senti.negative_score,
    ),
)


# Define senti result information
SentiResult.label.set_description(
    'Predicted label<br />For trinary model will be one of <br />"positive", "negative", "neutral";<br />For binary model will be one of <br />"positive", "negative"'
)
SentiResult.score.set_description("Probability of the predicted label: [0,1]")
SentiResult.scores.positive.set_description("Probability for positive prediction")
SentiResult.scores.neutral.set_description("Probability for neutral prediction (absent if use binary model)")
SentiResult.scores.negative.set_description("Probability for negative prediction")
