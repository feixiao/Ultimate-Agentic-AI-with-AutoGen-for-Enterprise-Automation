# Example code: Implementing GDPR right to explanation in an ML model

import shap


class ExplainableModel:
    """
    ExplainableModel wraps a machine learning model to provide GDPR-compliant explanations.
    """

    def __init__(self, model, feature_names):
        """
        Initialize the ExplainableModel with a trained model and its feature names.

        Parameters:
        * model: Trained machine learning model
        * feature_names: List of feature names
        """
        self.model = model  # Underlying machine learning model
        self.feature_names = feature_names  # Names of the features
        self.explainer = shap.TreeExplainer(
            model
        )  # SHAP explainer for tree-based models

    def predict(self, input_data):
        """
        Predict the output for the given input data.

        Parameters:
        * input_data: Data instance(s) for prediction

        Returns:
        * Model prediction
        """
        prediction = self.model.predict(input_data)
        return prediction

    def explain_prediction(self, input_data):
        """
        Generate a GDPR-compliant explanation for the model prediction.

        Parameters:
        * input_data: Data instance(s) to explain

        Returns:
        * Dictionary containing the prediction and key contributing factors
        """
        # Compute SHAP values for the input data
        shap_values = self.explainer.shap_values(input_data)

        # Initialize explanation dictionary
        explanation = {
            "prediction": self.model.predict(input_data)[0],  # First prediction value
            "factors_increasing_score": [],  # Features that increase the score
            "factors_decreasing_score": [],  # Features that decrease the score
        }

        # Pair feature names with their corresponding SHAP values and sort by impact
        feature_impacts = list(zip(self.feature_names, shap_values[0]))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        # Generate explanation using top 5 features
        for feature, impact in feature_impacts[:5]:
            if impact > 0:
                # Feature increases the prediction score
                explanation["factors_increasing_score"].append(
                    f"{feature}: Increased score by {impact:.2f} points"
                )
            else:
                # Feature decreases the prediction score
                explanation["factors_decreasing_score"].append(
                    f"{feature}: Decreased score by {abs(impact):.2f} points"
                )

        return explanation
