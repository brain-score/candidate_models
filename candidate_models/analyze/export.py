import json
import logging

import numpy as np
import pandas as pd
from pybtex.database.input import bibtex

from candidate_models.analyze import DataCollector

_logger = logging.getLogger(__name__)


def create_fixture(data=None):
    data = data or DataCollector()()
    data = data.dropna()
    fields = {'brain_score': 'brain-score', 'name': 'model', 'imagenet_top1': 'performance',
              'V4': 'V4', 'IT': 'IT', 'behavior': 'behavior',
              'V4_layer': 'V4-layer', 'IT_layer': 'IT-layer', 'behavior_layer': 'behavior-layer',
              'paper_link': 'link', 'paper_identifier': 'identifier'}

    def parse_bib(bibtex_str):
        bib_parser = bibtex.Parser()
        entry = bib_parser.parse_string(bibtex_str).entries
        assert len(entry) == 1
        entry = entry.values()[0]
        return entry

    bibs = data['bibtex'].apply(parse_bib)
    data['authors'] = bibs.apply(lambda entry: " ".join(entry.persons["author"][0].last()) + " et al.")
    data['year'] = bibs.apply(lambda entry: entry.fields['year'])
    data['identifier'] = data[['authors', 'year']].apply(lambda authors_year: ", ".join(authors_year), axis=1)
    data_rows = [
        {"model": "benchmarks.CandidateModel",
         "fields": {field_key: row[field] for field_key, field in fields.items()}}
        for _, row in data.iterrows()]
    savepath = 'fixture.json'
    _logger.info(f'Saving to {savepath}')
    with open(savepath, 'w') as f:
        json.dump(data_rows, f)


def highlight_max(data):
    assert data.ndim == 1
    try:
        is_max = data == data.max().tolist()
    except AttributeError:  # got a string (model)
        return data
    if isinstance(next(iter(data)), str):
        return [x.replace('_', '\\_') if isinstance(x, str) else x for x in data]
    all_ints = all(isinstance(x.tolist(), int) for x in data)
    data = ["{:.0f}".format(x) if all_ints else "{:.2f}".format(x) if x > 1 else "{:.03f}".format(x).lstrip('0')
            for x in data]  # format comma
    return [('\\textbf{' + str(x) + '}') if _is_max else x for x, _is_max in zip(data, is_max)]


def create_latex_table(data=None):
    data = data or DataCollector()()
    table = data.copy()
    aggregate_models = ['basenet', 'mobilenet']
    for model_prefix in aggregate_models:
        model_data = table[[model.startswith(model_prefix) for model in table['model']]]
        brainscores = model_data[model_data['benchmark'] == 'Brain-Score']
        best_model_index = brainscores['score'].astype(float).idxmax()
        best_model = brainscores.loc[best_model_index]
        table = table[[not model.startswith(model_prefix) or model == best_model['model'] for model in table['model']]]
        table['model'].loc[table['model'] == best_model['model']] = 'best ' + model_prefix

    benchmarks = {
        'Brain-Score': 'Brain-Score',
        'V4': 'dicarlo.Majaj2015.V4',
        'IT': 'dicarlo.Majaj2015.IT',
        'Behavior': 'dicarlo.Rajalingham2018',
        'ImageNet': 'ImageNet',
    }
    row_table = pd.DataFrame({'model': [model for model in np.unique(table['model'])
                                        if model not in ['cornet_z', 'cornet_r', 'cornet_r2']]})
    for key, benchmark in benchmarks.items():
        row_table[key] = [table[(table['benchmark'] == benchmark) & (table['model'] == model)]['score'].values.squeeze()
                          for model in row_table['model']]
    table = row_table[['Brain-Score', 'model', 'V4', 'IT', 'Behavior', 'ImageNet']]
    table = table.sort_values('Brain-Score', ascending=False)
    table = table.apply(highlight_max)
    table.to_latex('data.tex', escape=False, index=False)


if __name__ == '__main__':
    create_fixture()
