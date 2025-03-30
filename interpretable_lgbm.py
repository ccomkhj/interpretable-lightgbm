import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Optional, List
from datetime import datetime


class LGBM_SHAP_Explainer:
    """
    A class to make LightGBM models more interpretable using SHAP values.

    Parameters:
    -----------
    model : lightgbm.Booster or lightgbm.LGBMModel
        Trained LightGBM model
    train_data : pd.DataFrame or np.ndarray, optional
        Training data used to compute SHAP values (required for some explanations)
    feature_names : list or np.ndarray, optional
        List of feature names
    class_names : list, optional
        List of class names for classification problems
    """

    def __init__(
        self,
        model: Union[lgb.Booster, lgb.LGBMModel],
        train_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature_names: Optional[Union[List[str], np.ndarray]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.train_data = train_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type"""
        if isinstance(self.model, lgb.LGBMModel):
            if self.train_data is None:
                raise ValueError("Training data is required for LGBMModel explainer")
            return shap.Explainer(
                self.model.predict, self.train_data, feature_names=self.feature_names
            )
        else:
            return shap.TreeExplainer(self.model, feature_names=self.feature_names)

    def _get_feature_index(self, feature: Union[str, int]) -> int:
        """Helper method to get feature index from name or index"""
        if isinstance(feature, str):
            if self.feature_names is None:
                raise ValueError(
                    "Feature names not provided - cannot use string feature names"
                )

            # Convert feature_names to list if it's a numpy array
            feature_names_list = (
                list(self.feature_names)
                if isinstance(self.feature_names, np.ndarray)
                else self.feature_names
            )
            return feature_names_list.index(feature)
        return feature

    def summary_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        plot_type: str = "dot",
        max_display: int = 20,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Create SHAP summary plot showing feature importance.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        plot_type : str, default="dot"
            Type of plot ("dot", "violin", "bar")
        max_display : int, default=20
            Maximum number of features to display
        show : bool, default=True
            Whether to immediately display the plot
        save : bool, default=False
            Whether to save the plot to a file
        filename : str, optional
            Filename for saving (default: "summary_plot_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to shap.summary_plot
        """
        shap_values = self.explainer(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            plot_type=plot_type,
            feature_names=self.feature_names,
            class_names=self.class_names,
            max_display=max_display,
            show=False,
            **kwargs,
        )

        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"shap_summary_plot_{timestamp}.jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def feature_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        plot: bool = True,
        max_display: int = 20,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute and optionally plot SHAP feature importance.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to compute importance on
        plot : bool, default=True
            Whether to plot the importance
        max_display : int, default=20
            Maximum number of features to display
        show : bool, default=True
            Whether to display the plot
        save : bool, default=False
            Whether to save the plot
        filename : str, optional
            Filename for saving (default: "feature_importance_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to plotting function

        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importance values
        """
        shap_values = self.explainer(X)

        if len(shap_values.shape) == 3:  # For multi-class
            vals = np.abs(shap_values.values).mean(0)
        else:
            vals = np.abs(shap_values.values).mean(0)

        if self.feature_names is None:
            feature_names = [f"f{i}" for i in range(len(vals))]
        else:
            # Convert feature_names to list if it's a numpy array
            feature_names = (
                list(self.feature_names)
                if isinstance(self.feature_names, np.ndarray)
                else self.feature_names
            )

        importance_df = pd.DataFrame(
            {"feature": feature_names, "shap_importance": vals}
        ).sort_values("shap_importance", ascending=False)

        if plot:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=max_display, show=False)
            plt.title("SHAP Feature Importance")

            if save:
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"shap_feature_importance_{timestamp}.jpg"
                plt.savefig(filename, bbox_inches="tight", dpi=300)
            if show:
                plt.show()
            else:
                plt.close()

        return importance_df

    def dependence_plot(
        self,
        feature: Union[str, int],
        X: Union[pd.DataFrame, np.ndarray],
        interaction_index: Optional[Union[str, int]] = None,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Create SHAP dependence plot for a single feature.

        Parameters:
        -----------
        feature : str or int
            Feature name or index to analyze. Shows how the model output changes
            as this feature varies.
        X : pd.DataFrame or np.ndarray
            Data to explain
        interaction_index : str or int, optional
            Feature to use for coloring. Shows potential interaction effects.
        show : bool, default=True
            Whether to display the plot
        save : bool, default=False
            Whether to save the plot
        filename : str, optional
            Filename for saving (default: "dependence_plot_{feature}_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to shap.dependence_plot
        """
        shap_values = self.explainer(X).values

        # Get feature index using helper method
        feature_index = self._get_feature_index(feature)

        # Handle interaction index if provided
        if interaction_index is not None:
            interaction_index = self._get_feature_index(interaction_index)

        plt.figure(figsize=(12, 8))
        shap.dependence_plot(
            ind=feature_index,
            shap_values=shap_values,
            features=X,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False,
            **kwargs,
        )
        plt.title(f"SHAP Dependence Plot for '{feature}'")

        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                feature_name = (
                    feature if isinstance(feature, str) else f"feature_{feature}"
                )
                filename = f"shap_dependence_{feature_name}_{timestamp}.jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def force_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: int = 0,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Create SHAP force plot for a single instance.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        instance_index : int, default=0
            Positional index (0-based) of the instance in X to explain.
            For DataFrames, this corresponds to X.iloc[instance_index].
            The plot shows how each feature contributes to pushing the
            model output from the base value to the final prediction.
        show : bool, default=True
            Whether to display the plot
        save : bool, default=False
            Whether to save the plot
        filename : str, optional
            Filename for saving (default: "force_plot_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to shap.force_plot
        """
        shap_values = self.explainer(X)

        if hasattr(shap_values, "base_values"):
            expected_value = shap_values.base_values[instance_index]
        else:
            if isinstance(self.model, lgb.LGBMModel):
                expected_value = self.model.predict(X).mean()
            else:
                expected_value = self.model.predict(
                    X, num_iteration=self.model.best_iteration
                ).mean()

        if len(shap_values.shape) == 3:  # Multi-class
            class_index = 0
            shap_values_instance = shap_values[instance_index, :, class_index]
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[class_index]
        else:
            shap_values_instance = shap_values[instance_index]

        if hasattr(X, "iloc"):
            features = X.iloc[instance_index]
        else:
            features = X[instance_index, :]

        plt.figure()
        shap.force_plot(
            base_value=expected_value,
            shap_values=(
                shap_values_instance.values
                if hasattr(shap_values_instance, "values")
                else shap_values_instance
            ),
            features=features,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
            **kwargs,
        )

        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"shap_force_plot_{timestamp}.jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def decision_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_indices: Union[int, List[int]] = 0,
        class_index: Optional[int] = None,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Create SHAP decision plot for one or more instances.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        instance_indices : int or list of int, default=0
            Positional indices (0-based) of instances in X to explain.
            For DataFrames, corresponds to X.iloc[instance_indices].
            The plot shows how each feature contributes to the final prediction.
        class_index : int, optional
            For multi-class problems, which class to explain (default: first class)
        show : bool, default=True
            Whether to display the plot
        save : bool, default=False
            Whether to save the plot
        filename : str, optional
            Filename for saving (default: "decision_plot_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to shap.decision_plot
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X)
        else:
            shap_values = self.explainer(X).values

        if isinstance(instance_indices, int):
            instance_indices = [instance_indices]

        if hasattr(self.explainer, "expected_value"):
            expected_value = self.explainer.expected_value
        else:
            if isinstance(self.model, lgb.LGBMModel):
                expected_value = self.model.predict(X).mean()
            else:
                expected_value = self.model.predict(
                    X, num_iteration=self.model.best_iteration
                ).mean()

        if isinstance(shap_values, list):
            if class_index is None:
                class_index = 0
            shap_values_subset = shap_values[class_index][instance_indices]
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[class_index]
        else:
            shap_values_subset = shap_values[instance_indices]

        X_subset = (
            X.iloc[instance_indices] if hasattr(X, "iloc") else X[instance_indices, :]
        )

        plt.figure(figsize=(10, 6))
        shap.decision_plot(
            expected_value,
            shap_values_subset,
            X_subset,
            feature_names=self.feature_names,
            ignore_warnings=True,
            show=False,
            **kwargs,
        )

        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"shap_decision_plot_{timestamp}.jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def waterfall_plot(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_index: int = 0,
        class_index: Optional[int] = None,
        show: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Create SHAP waterfall plot for a single instance.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Data to explain
        instance_index : int, default=0
            Positional index (0-based) of the instance in X to explain.
            For DataFrames, corresponds to X.iloc[instance_index].
            The plot shows how each feature contributes to the final prediction,
            starting from the expected value.
        class_index : int, optional
            For multi-class problems, which class to explain (default: first class)
        show : bool, default=True
            Whether to display the plot
        save : bool, default=False
            Whether to save the plot
        filename : str, optional
            Filename for saving (default: "waterfall_plot_{timestamp}.jpg")
        **kwargs : dict
            Additional arguments passed to shap.plots.waterfall
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X)
        else:
            shap_values = self.explainer(X).values

        if hasattr(self.explainer, "expected_value"):
            expected_value = self.explainer.expected_value
        else:
            if isinstance(self.model, lgb.LGBMModel):
                expected_value = self.model.predict(X).mean()
            else:
                expected_value = self.model.predict(
                    X, num_iteration=self.model.best_iteration
                ).mean()

        if isinstance(shap_values, list):
            if class_index is None:
                class_index = 0
            shap_values_instance = shap_values[class_index][instance_index]
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[class_index]
        else:
            shap_values_instance = shap_values[instance_index]

        explanation = shap.Explanation(
            values=shap_values_instance,
            base_values=expected_value,
            data=X[instance_index] if hasattr(X, "iloc") else X[instance_index, :],
            feature_names=self.feature_names,
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False, **kwargs)

        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"shap_waterfall_{timestamp}.jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load data and train a model
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    # Create explainer
    explainer = LGBM_SHAP_Explainer(model, X_train, feature_names=feature_names)

    # Generate various explanations (with save examples)
    # explainer.summary_plot(X_test, save=True)
    # explainer.feature_importance(X_test, save=True)
    # explainer.dependence_plot("worst radius", X_test, save=True)
    # explainer.force_plot(X_test, instance_index=0, save=True)
    explainer.decision_plot(X_test, instance_indices=[0, 1, 2], save=True)
    explainer.waterfall_plot(X_test, instance_index=0, save=True)
