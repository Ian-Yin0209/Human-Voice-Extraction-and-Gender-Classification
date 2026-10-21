"""Convert the original TensorFlow variable layout to U-Net state dictionaries."""

from pathlib import Path

import numpy as np


def tf2pytorch(checkpoint_path, num_instrumments):
    """Read a TensorFlow checkpoint prefix, reporting incomplete assets clearly."""
    prefix = Path(checkpoint_path)
    if not Path(str(prefix) + ".index").is_file():
        raise FileNotFoundError(
            f"TensorFlow checkpoint index not found: {prefix}.index. "
            "Pass the checkpoint prefix, for example checkpoints/2stems/model."
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Reading TensorFlow checkpoints requires model dependencies. "
            "Install them from the repository with: pip install '.[models]'"
        ) from exc

    tf_vars = {}
    try:
        for name, _ in tf.train.list_variables(str(prefix)):
            tf_vars[name] = tf.train.load_variable(str(prefix), name)
    except Exception as exc:
        raise RuntimeError(f"Unable to read TensorFlow checkpoint {prefix}: {exc}") from exc
    try:
        return _convert_weights(tf_vars, num_instrumments)
    except KeyError as exc:
        raise ValueError(f"Checkpoint {prefix} is missing required variable {exc}.") from exc


def _convert_weights(tf_vars, num_instrumments):
    conv_idx = 0
    tconv_idx = 0
    bn_idx = 0
    outputs = []
    for i in range(num_instrumments):
        output = {}
        outputs.append(output)

        for j in range(1,7):
            if conv_idx == 0:
                conv_suffix = ""
            else:
                conv_suffix = "_" + str(conv_idx)

            if bn_idx == 0:
                bn_suffix = ""
            else:
                bn_suffix = "_" + str(bn_idx)

            output['down{}_conv.weight'.format(j)] = np.transpose(
                tf_vars["conv2d{}/kernel".format(conv_suffix)], (3, 2, 0, 1))
            # print('conv dtype: ',output['down{}.0.weight'.format(j)].dtype)
            output['down{}_conv.bias'.format(
                j)] = tf_vars["conv2d{}/bias".format(conv_suffix)]

            output['down{}_act.0.weight'.format(
                j)] = tf_vars["batch_normalization{}/gamma".format(bn_suffix)]
            output['down{}_act.0.bias'.format(
                j)] = tf_vars["batch_normalization{}/beta".format(bn_suffix)]
            output['down{}_act.0.running_mean'.format(
                j)] = tf_vars['batch_normalization{}/moving_mean'.format(bn_suffix)]
            output['down{}_act.0.running_var'.format(
                j)] = tf_vars['batch_normalization{}/moving_variance'.format(bn_suffix)]

            conv_idx += 1
            bn_idx += 1

        # up blocks
        for j in range(1, 7):
            if tconv_idx == 0:
                tconv_suffix = ""
            else:
                tconv_suffix = "_" + str(tconv_idx)

            if bn_idx == 0:
                bn_suffix = ""
            else:
                bn_suffix= "_" + str(bn_idx)

            output['up{}.0.weight'.format(j)] = np.transpose(
                tf_vars["conv2d_transpose{}/kernel".format(tconv_suffix)], (3,2,0, 1))
            output['up{}.0.bias'.format(
                j)] = tf_vars["conv2d_transpose{}/bias".format(tconv_suffix)]
            output['up{}.2.weight'.format(
                j)] = tf_vars["batch_normalization{}/gamma".format(bn_suffix)]
            output['up{}.2.bias'.format(
                j)] = tf_vars["batch_normalization{}/beta".format(bn_suffix)]
            output['up{}.2.running_mean'.format(
                j)] = tf_vars['batch_normalization{}/moving_mean'.format(bn_suffix)]
            output['up{}.2.running_var'.format(
                j)] = tf_vars['batch_normalization{}/moving_variance'.format(bn_suffix)]
            tconv_idx += 1
            bn_idx += 1

        if conv_idx == 0:
            suffix = ""
        else:
            suffix = "_" + str(conv_idx)
        output['up7.0.weight'] = np.transpose(
            tf_vars['conv2d{}/kernel'.format(suffix)], (3, 2, 0, 1))
        output['up7.0.bias'] = tf_vars['conv2d{}/bias'.format(suffix)]
        conv_idx += 1

    return outputs
