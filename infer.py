from pathlib import Path

from jokes_mlops_project import model_from_json


def main():
    # load model
    model_path = Path("chekpoints") / "model.json"
    language_model = model_from_json(model_path)

    # generate
    prefixes = ["заходит мужик в бар", "купил мужик шляпу", "идёт медведь по лесу"]
    for prefix in prefixes:
        print(language_model.generate(prefix))


if __name__ == "__main__":
    main()
