import json
import tensorflow as tf
import os
import logging
import requests
import tarfile

from unsup_vvs.neural_fit.cleaned_network_builder import get_network_outputs

from model_tools.activations.tensorflow import TensorflowSlimWrapper
from model_tools.activations.tensorflow import load_resize_image

TF_RES18_LAYERS = ['encode_1.conv'] + ['encode_%i' % i for i in range(1, 10)]
_logger = logging.getLogger(__name__)


class ModelBuilder:
    CKPT_PATH = {
        'resnet18-supervised': 'http://visualmaster-models.s3.amazonaws.com/supervised/seed0/checkpoint-505505.tar',
        'resnet18-la': 'http://visualmaster-models.s3.amazonaws.com/la/seed1/checkpoint-2502500.tar',
        'resnet18-ir': 'http://visualmaster-models.s3.amazonaws.com/ir/seed1/checkpoint-2502500.tar',
        'resnet18-ae': 'http://visualmaster-models.s3.amazonaws.com/ae/seed0/checkpoint-1301300.tar',
        'resnet18-cpc': 'http://visualmaster-models.s3.amazonaws.com/cpc/seed0/model.ckpt-1301300.tar',
        'resnet18-color': 'http://visualmaster-models.s3.amazonaws.com/color/seed0/model.ckpt-5605040.tar',
    }
    PREP_TYPE = {
        'resnet18-color': 'color_prep',
        'resnet18-depth': 'no_prep',
        'resnet18-rp': 'only_mean',
    }
    CFG_KWARGS = {
        'resnet18-color': {
            'ignorebname_new': 0,
            'add_batchname': '_colorization'},
        'resnet18-depth': {
            'ignorebname_new': 0,
            'add_batchname': '_pbrnet'},
        'resnet18-rp': {
            'ignorebname_new': 0,
            'add_batchname': '_rp'},
    }

    def __call__(self, identifier):
        if identifier not in self.CKPT_PATH:
            raise ValueError(f"No known checkpoint for identifier {identifier}")
        load_from_ckpt = self.__get_ckpt_from_aws(identifier)
        return self.__get_tf_model(
                identifier=identifier, load_from_ckpt=load_from_ckpt,
                prep_type=self.PREP_TYPE.get(identifier, 'mean_std'),
                cfg_kwargs=self.CFG_KWARGS.get(identifier, {}))

    def __get_ckpt_from_aws(self, identifier):
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weightsdir_path = os.path.join(framework_home, 'model-weights', 'unsup_vvs', identifier)
        aws_path = self.CKPT_PATH[identifier]
        weight_data_name = os.path.basename(aws_path)[:-3] + 'data-00000-of-00001'
        weights_path = os.path.join(weightsdir_path, weight_data_name)
        if os.path.exists(weights_path):
            _logger.debug(f"Using cached weights at {weights_path}")
        else:
            _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
            os.makedirs(weightsdir_path, exist_ok=True)
            tar_path = os.path.join(
                    weightsdir_path, os.path.basename(aws_path))
            r = requests.get(aws_path, allow_redirects=True)
            with open(tar_path, 'wb') as tar_file:
                tar_file.write(r.content)
            tar = tarfile.open(tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=weightsdir_path)
            os.remove(tar_path)
        return os.path.join(weightsdir_path, os.path.basename(aws_path)[:-4])

    def __get_tf_model(self,
                       identifier, load_from_ckpt,
                       batch_size=64, model_type='vm_model', prep_type='mean_std',
                       cfg_kwargs={}):
        img_path_placeholder = tf.placeholder(
            dtype=tf.string,
            shape=[batch_size])
        ending_points = self._build_model_ending_points(
            img_paths=img_path_placeholder, prep_type=prep_type, 
            model_type=model_type, cfg_kwargs=cfg_kwargs)

        SESS = self.get_tf_sess_restore_model_weight(load_from_ckpt=load_from_ckpt)

        self.ending_points = ending_points
        self.img_path_placeholder = img_path_placeholder
        self.SESS = SESS
        self.identifier = identifier
        return self._build_activations_model(batch_size=batch_size)

    def _build_model_ending_points(self, img_paths, prep_type, model_type,
                                   setting_name='cate_res18_exp0', cfg_kwargs={}):
        imgs = self._get_imgs_from_paths(img_paths)

        ending_points, _ = get_network_outputs(
            {'images': imgs},
            prep_type=prep_type,
            model_type=model_type,
            setting_name=setting_name,
            **cfg_kwargs)
        for key in ending_points:
            if len(ending_points[key].get_shape().as_list()) == 4:
                ending_points[key] = tf.transpose(
                    ending_points[key],
                    [0, 3, 1, 2])
        return ending_points

    def _get_imgs_from_paths(self, img_paths):
        _load_func = lambda image_path: load_resize_image(
            image_path, 224)
        imgs = tf.map_fn(_load_func, img_paths, dtype=tf.float32)
        return imgs

    def get_tf_sess_restore_model_weight(self, load_var_list=None, from_scratch=False, load_from_ckpt=None):
        SESS = self.get_tf_sess()
        if load_var_list is not None:
            name_var_list = json.loads(load_var_list)
            needed_var_list = {}
            curr_vars = tf.global_variables()
            curr_names = [variable.op.name for variable in curr_vars]
            for old_name in name_var_list:
                new_name = name_var_list[old_name]
                assert new_name in curr_names, "Variable %s not found!" % new_name
                _ts = curr_vars[curr_names.index(new_name)]
                needed_var_list[old_name] = _ts
            saver = tf.train.Saver(needed_var_list)

            init_op_global = tf.global_variables_initializer()
            SESS.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            SESS.run(init_op_local)
        else:
            saver = tf.train.Saver()

        if not from_scratch:
            saver.restore(SESS, load_from_ckpt)
        else:
            init_op_global = tf.global_variables_initializer()
            SESS.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            SESS.run(init_op_local)

        assert len(SESS.run(tf.report_uninitialized_variables())) == 0, \
            (SESS.run(tf.report_uninitialized_variables()))
        return SESS

    def get_tf_sess(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        SESS = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
        ))
        return SESS

    def _build_activations_model(self, batch_size):
        self.activations_model = TensorflowSlimWrapper(
            identifier=self.identifier, labels_offset=0,
            endpoints=self.ending_points,
            inputs=self.img_path_placeholder,
            session=self.SESS,
            batch_size=batch_size)
        return self.activations_model
