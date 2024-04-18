from seaborn import load_dataset
from julearn import PipelineCreator, run_cross_validation
from julearn.utils import configure_logging
from sklearn.model_selection import StratifiedKFold
from julearn.viz import plot_scores


def main():
    configure_logging(level="INFO")

    df_iris = load_dataset("iris")

    df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

    X = ["sepal_length", "sepal_width", "petal_length"]

    y = "species"

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    creator = PipelineCreator(problem_type="classification")
    creator.add("zscore")
    creator.add("svm", kernel=["linear", 'rbf'], C=[0.1, 1, 10])

    scores, model, inspector = run_cross_validation(
        X=X,
        y=y,
        data=df_iris,
        cv=cv,
        model=creator,
        return_train_score=True,
        return_inspector=True,
    )

    print(scores)
    print(inspector)
    scores.to_csv("scores/scores.csv", index=False)
    panel = plot_scores(scores)
    panel.show()


if __name__ == "__main__":
    main()
