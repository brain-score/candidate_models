from candidate_models.analyze import DataCollector, filter_basenets


def best_models():
    data = DataCollector()()

    def best(data, criterion):
        index = data[criterion].idxmax()
        best_model = data.loc[index]
        print(f"Best {criterion}: {best_model['model']}, brain-score {best_model[criterion]}")

    for criterion in ['brain-score', 'V4', 'IT', 'behavior']:
        best(data, criterion)

    print()
    print("Basenets")
    basenets = filter_basenets(data)
    for criterion in ['brain-score', 'V4', 'IT', 'behavior']:
        best(basenets, criterion)

    print()
    print("CORnet")
    cornet = data[[row['model'].startswith('cornet') for _, row in data.iterrows()]]
    for criterion in ['brain-score', 'V4', 'IT', 'behavior']:
        best(cornet, criterion)


if __name__ == '__main__':
    best_models()
