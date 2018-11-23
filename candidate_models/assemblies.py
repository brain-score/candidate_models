import os
from glob import glob

import pandas as pd

import brainscore
from brainscore.stimuli import StimulusSet

gallant_dir = os.path.join(os.path.dirname(__file__), '..', 'mkgu_packaging', 'mkgu_packaging', 'gallant')


def load_stimulus_set(stimulus_set_name):
    if stimulus_set_name == 'objectome-iclr':
        paths = list(glob('/braintree/data2/active/common/objectome64s100/*.png'))
        basenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        image_ids = [os.path.splitext(name)[0] for name in basenames]
        s = StimulusSet({'image_file_path': paths, 'image_file_name': basenames, 'image_id': image_ids})
        s.image_paths = {image_id: path for image_id, path in zip(image_ids, paths)}
        return s
    if stimulus_set_name.startswith('objectome'):
        # TODO: remove once packaged in brainscore
        directory = os.path.join(os.path.dirname(__file__), '..', 'mkgu_packaging',
                                 stimulus_set_name, stimulus_set_name + '-224')
        return _mock_stimulus_set(directory, extension='png')
    if stimulus_set_name.startswith('cifar'):
        stimulus_set = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', stimulus_set_name + '.pkl'))['data']
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.image_id: row.image_file_path for row in stimulus_set.itertuples()}
        return stimulus_set
    if stimulus_set_name == 'coco':
        stimulus_set = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', stimulus_set_name + '.pkl'))['data']
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.image_id: row.image_file_path for row in stimulus_set.itertuples()}
        return stimulus_set
    # if stimulus_set_name == 'gallant.David2004':
    #     directory = os.path.join(gallant_dir, 'V1Data/NatRev/stimuli/stimuli-{}'.format(224))
    #     return _mock_stimulus_set(directory, extension='jpg')
    if stimulus_set_name == 'gallant.Willmore2010':
        directory = os.path.join(os.path.dirname(__file__), '..', 'mkgu_packaging', 'mkgu_packaging',
                                 'gallant/V2Data/NatRev/stimuli/stimuli-{}'.format(224))
        return _mock_stimulus_set(directory, extension='jpg')
    stimulus_set = brainscore.get_stimulus_set(stimulus_set_name)
    degrees = {'tolias.Cadena2017': 2,
               'movshon.FreemanZiemba2013': 4,
               'dicarlo.hvm': 8}
    if stimulus_set_name in degrees:
        stimulus_set['degrees'] = degrees[stimulus_set_name]
    return stimulus_set


def _mock_stimulus_set(directory, extension='png'):
    s = StimulusSet(pd.read_csv(os.path.join(directory, 'stimulus_set.csv')))
    s.image_paths = {image_id: os.path.join(directory, image_id + '.' + extension) for image_id in s['image_id']}
    return s
