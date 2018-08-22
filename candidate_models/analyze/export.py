import json
import logging

from pybtex.database.input import bibtex

from candidate_models.analyze import DataCollector

_logger = logging.getLogger(__name__)


def create_fixture(data=None):
    data = data or DataCollector()()
    data = data.dropna()
    fields = {'brain_score': 'brain-score', 'name': 'model', 'imagenet_top1': 'performance',
              'v4': 'V4', 'it': 'IT', 'behavior': 'behavior',
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
    is_max = data == data.max()
    if isinstance(next(iter(data)), str):
        return [x.replace('_', '\\_') if isinstance(x, str) else x for x in data]
    all_ints = all(isinstance(x, int) or x.is_integer() for x in data)
    data = ["{:.0f}".format(x) if all_ints else "{:.2f}".format(x) if x > 1 else "{:.03f}".format(x).lstrip('0')
            for x in data]  # format comma
    return [('\\textbf{' + str(x) + '}') if _is_max else x for x, _is_max in zip(data, is_max)]


def create_latex_table(data=None):
    data = data or DataCollector()()
    table = data[[not model.startswith('basenet') for model in data['model']]]
    table = table[['brain-score', 'model', 'performance'] +
                  ['V4', 'IT', 'behavior']]
    table = table.sort_values('brain-score', ascending=False)
    table = table.rename(columns={'brain-score': 'Brain Score'})
    table = table.apply(highlight_max)
    table.to_latex('data.tex', escape=False, index=False)
