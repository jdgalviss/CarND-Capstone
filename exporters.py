import tensorflow as tf
import time
import os
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import shutil
import subprocess
import hashlib

DEFAULT_TAGS = [tf.saved_model.tag_constants.SERVING]
DEFAULT_SERVING_SIGNATURE_DEF_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
SAVED_MODEL_MAIN_OP = "saved_model_main_op"


def cache_saved_model(export_dir, rm=False, save_path=None):
    r"""Caches a SavedModel on a distributed storage to a local path. If `export_dir` starts with `s3://` or `gs://` it will download a local copy of the saved model. To use this a proper install and credentials setup of `aws-cli` or `gsutil` have to be inplace.
    Parameters
    ----------
    export_dir : str
        The path where the SavedModel is stored.
    cache: bool, default = False
        If `cache` is `True` and .
    
    Returns
    -------
    str
        The path to the local version of the SavedModel. If `export_dir` was already a local path it will immediatly return it instead.
    """

    if not export_dir.startswith("s3://") and export_dir.startswith("s3://"):
        return export_dir

    hash_name = str(hashlib.sha256(export_dir.encode()).hexdigest())

    if save_path:
        model_dir_base = save_path
    else:
        home_path = os.path.expanduser('~')
        model_dir_base = os.path.join(home_path, ".local", "avi_model_tools", "saved_models")

    model_dir_base = os.path.join(model_dir_base, hash_name)

    if os.path.exists(model_dir_base):

        is_empty = len(os.listdir(model_dir_base)) == 0
        files_counts = [os.path.join(dp, f) for dp, _, filenames in os.walk(model_dir_base) for f in filenames if f.endswith(".gstmp")]
        temp_files = len(files_counts) > 0

        if rm or is_empty or temp_files:
            shutil.rmtree(model_dir_base, ignore_errors=True)
        else:
            return model_dir_base

    if not os.path.exists(model_dir_base):
        os.makedirs(model_dir_base)

        if export_dir.startswith("s3://"):
            cp_cmd = "aws s3 cp --recursive"
        elif export_dir.startswith("gs://"):
            cp_cmd = "gsutil -m cp -R"
        else:
            cp_cmd = "cp -R"

        cmd = "{cp_cmd} {source_folder} {dest_folder}".format(
            cp_cmd=cp_cmd,
            source_folder=export_dir,
            dest_folder=model_dir_base,
        )

        subprocess.call(
            cmd,
            stdout=subprocess.PIPE,
            shell=True,
        )

        return model_dir_base


def load_saved_model(session, export_dir, input_map, scope=None, tags=None, signature_def_key=None, cache=False, return_inputs=False, **saver_kwargs):
    r"""Loads the graph of a `SavedModel` on `export_dir` into the give `session`. 
    Parameters
    ----------
    session : Session
        The session where to load the SavedModel into.
    export_dir : str
        The path where the SavedModel is stored.
    input_map : Dict[str, Tensor]
        The parameter `input_map` is a dictionary from `str` to `Tensor` that lets you replace the input nodes/tensors of the SavedModel graph by you own Tensor, this lets you pass you stich models; the key of `input_map` has to be the real tensor name as defined in the original graph, not the `signature_def` created when exporting.
    tags: List[str], optional
        SavedModel targs
    signature_def_key: str, optional
        SavedModel signature key def.
    cache: bool, default = False
        If `cache` is `True` and `export_dir` is and `s3://` or `gs://` path  it will download a local copy of the saved model and use it, for this proper install and credentials to `aws-cli` or `gsutil` have to be inplace. If your model is on a distributed storage and this is not used, the model will have to be downloaded by TF every time. Uses the `cache_saved_model` function internally.
    return_inputs: bool, default = False
        Whether to return the inputs as well as the outputs.
    Returns
    -------
    Dict[str, Tensor] | Tuple[Dict[str, Tensor]]
        A dicctionary of `signature_def.outputs` names to their corresponding `Tensor`s in the loaded `Graph`. If `return_inputs` is `True` then it will return a tuple `(inputs, outputs)`. 
    """

    if cache:
        export_dir = cache_saved_model(export_dir)

    if tags is None:
        tags = DEFAULT_TAGS

    if signature_def_key is None:
        signature_def_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY

    saver_kwargs["input_map"] = input_map

    metagraph_def = tf.saved_model.loader.load(
        session,
        tags,
        export_dir,
        import_scope=scope,
        **saver_kwargs
    )
    signature_def = metagraph_def.signature_def[signature_def_key]

    inputs = {name: session.graph.get_tensor_by_name(v.name if scope is None else os.path.join(scope, v.name)) for name, v in signature_def.inputs.items()}

    outputs = {name: session.graph.get_tensor_by_name(v.name if scope is None else os.path.join(scope, v.name)) for name, v in signature_def.outputs.items()}

    if return_inputs:
        return inputs, outputs
    else:
        return outputs


def simple_save(
        session,
        export_dir,
        inputs,
        outputs,
        timestamp=True,
        as_frozen_graph=False,
        optimize=False,
        print_msg=False,
        export_graph=False,
):
    r"""A wrapper around `tf.saved_model.simple_save` that additionally merge the ops in the `saved_model_main_op` collection into a single `main_op`; does some book keeping around this collection to leave it as is.
    Parameters
    ----------
    session : Session
        The session where holding the Tensors to save.
    export_dir : str
        The path where SavedModel to serialize the `SavedModel`
    inputs: Dict[str, Tensor]
        See `tf.saved_model.simple_save`
    outputs: Dict[str, Tensor]
        See `tf.saved_model.simple_save`
    timestamp: bool, default = True
        Whether or not to create the saved model on a folder whose name is a timestamp.
    as_frozen_graph: bool, default = False
        If the model should be serialized as a frozen graph instead of a SavedModel
    optimize: bool, default = False
        If the model's graph shoud be optimized before serialization, optimization includes operations like `strip_unused`, `remove_training_nodes`, `fold_batch_norms`, `fuse_resize_and_conv`.
    print_msg: bool, default = False
        If `True` it will print a summary message.
    export_graph: bool, default = False
        Export graph summary of the output model so it can be visualized in Tensorboard. 
    Returns
    -------
    str
        A string with the output path of the model.
    """

    input_names = {key: tensor.name.split(":")[0] for key, tensor in inputs.items()}
    output_names = {key: tensor.name.split(":")[0] for key, tensor in outputs.items()}
    tensor_input_names = list(input_names.values())
    tensor_output_names = list(output_names.values())

    if optimize:

        input_dtypes = [tensor.dtype.as_datatype_enum for tensor in inputs.values()]
        input_graph_def = session.graph.as_graph_def()

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            tensor_input_names,
            tensor_output_names,
            input_dtypes,
            toco_compatible=False,
        )

        with tf.Graph().as_default(), tf.Session() as session:

            tf.import_graph_def(
                output_graph_def,
                name="",
            )

            graph = session.graph

            inputs = {key: graph.get_tensor_by_name(tensor.name) for key, tensor in inputs.items()}
            outputs = {key: graph.get_tensor_by_name(tensor.name) for key, tensor in outputs.items()}

            # tensor_input_values = nodes[:n_inputs]
            # tensor_output_values = nodes[n_inputs:]

            # inputs = dict(zip(tensor_input_names, tensor_input_values))
            # outputs = dict(zip(tensor_output_names, tensor_output_values))

            return simple_save(
                session=session,
                export_dir=export_dir,
                inputs=inputs,
                outputs=outputs,
                optimize=False,
                timestamp=timestamp,
                as_frozen_graph=as_frozen_graph,
                print_msg=print_msg,
                export_graph=export_graph,
            )

    main_ops = session.graph.get_collection(SAVED_MODEL_MAIN_OP)
    session.graph.clear_collection(SAVED_MODEL_MAIN_OP)

    main_op = tf.group(*main_ops)
    session.graph.add_to_collection(SAVED_MODEL_MAIN_OP, main_op)

    if timestamp:
        output_path = os.path.join(export_dir, _timestamp())
    else:
        output_path = export_dir

    if as_frozen_graph:
        tf.gfile.MakeDirs(output_path)
        output_path = os.path.join(output_path, "model.pb")
        graph_def = session.graph.as_graph_def()

        graph_def = tf.graph_util.remove_training_nodes(graph_def)
        graph_def = tf.graph_util.convert_variables_to_constants(
            session,
            graph_def,
            tensor_output_names,
        )

        with tf.gfile.GFile(output_path, "wb") as f:
            f.write(graph_def.SerializeToString())
    else:
        tf.saved_model.simple_save(
            session=session,
            export_dir=output_path,
            inputs=inputs,
            outputs=outputs,
            # legacy_init_op=tf.group(*main_ops),
        )

        # for op in main_ops:
        #     session.graph.add_to_collection(SAVED_MODEL_MAIN_OP, op)

    if print_msg:
        print(model_message(inputs, outputs, output_path, tensorboard=export_graph))

    if export_graph:
        export_current_graph(output_path, session)

    return output_path


def get_signature(export_dir, cache=False):
    r"""Gets the signature def of the SavedModel, it will not load the tensors to the currect graph. 
    Parameters
    ----------
    export_dir : str
        The path where the SavedModel is stored.
    cache: bool, default = False
        If `cache` is `True` and `export_dir` is and `s3://` or `gs://` path  it will download a local copy of the saved model and use it, for this proper install and credentials to `aws-cli` or `gsutil` have to be inplace. If your model is on a distributed storage and this is not used, the model will have to be downloaded by TF every time. Uses the `cache_saved_model` function internally.
    
    Returns
    -------
    Tuple[Dict[str, Tensor]]
        This function will return a tuple `(inputs, outputs)` where inputs and outputs are the dictionaries of the SavedModel signature def.
    """
    with tf.Graph().as_default(), tf.Session().as_default() as sess_tmp:

        return load_saved_model(
            sess_tmp,
            export_dir,
            {},
            scope="",
            cache=cache,
            return_inputs=True,
        )


def model_message(inputs, outputs, export_dir, tensorboard=False):

    if tensorboard:
        if export_dir.endswith(".pb"):
            tensorboard_dir = os.path.dirname(export_dir)
        else:
            tensorboard_dir = export_dir

        tensorboard_msg = "  tensorboard --logdir " + tensorboard_dir + " --host 0.0.0.0"
        tensorboard_msg = "TENSORBOARD\n" + tensorboard_msg + "\n\n"
    else:
        tensorboard_msg = ""

    msg_input_title = "INPUTS"
    msg_inputs = ["- {key} : {tensor}".format(key=key, tensor=tensor) for key, tensor in inputs.items()]
    msg_output_title = "OUTPUTS"
    msg_outputs = ["- {key} : {tensor}".format(key=key, tensor=tensor) for key, tensor in outputs.items()]
    msg_save = "EXPORT_DIR\n  {export_dir}".format(export_dir=export_dir)
    # msg_separator = "#" * max(len(msg_output_title), len(msg_output_title), len(msg_save), *list(map(len, msg_inputs)), *list(map(len, msg_outputs)))
    full_msg = "{msg_separator}\n## EXPORT SUMMARY\n{msg_separator}\n\n{msg_input_title}\n{msg_inputs}\n\n{msg_output_title}\n{msg_outputs}\n\n{msg_save}\n\n{tensorboard_msg}{msg_separator}".format(
        msg_separator=msg_separator,
        msg_input_title=msg_input_title,
        msg_inputs="\n".join(msg_inputs),
        msg_output_title=msg_output_title,
        msg_outputs="\n".join(msg_outputs),
        msg_save=msg_save,
        tensorboard_msg=tensorboard_msg,
    )

    return full_msg


def export_current_graph(export_dir, session):

    if export_dir.endswith(".pb"):
        export_dir = os.path.dirname(export_dir)

    _writer = tf.summary.FileWriter(
        logdir=export_dir,
        graph=session.graph,
    )


def _timestamp():
    return str(int(time.time()))


from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

def to_frozen_graph(export_dir, output_path):
    r"""Creates a frozen graph from a Saved Model.
    Parameters
    ----------
    --export-dir: str
        Path of the Saved Model directory.
    --output-path: str
        Path for the output frozen graph protobuf.
    """

    with tf.Session() as session:
        print("### Getting input and output node names -> ###")

        meta_graph_def = tf.saved_model.loader.load(
            session,
            DEFAULT_TAGS,
            export_dir,
        )

        signature = meta_graph_def.signature_def

        in_tensor_names = [t.name for t in signature[DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs.values()]
        out_tensor_names = [t.name for t in signature[DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs.values()]

        # os.makedirs(
        #     os.path.dirname(output_path),
        #     exist_ok=True,
        # )

        freeze_graph.freeze_graph(
            input_graph=None,
            input_saver=None,
            input_binary=False,
            input_checkpoint=None,
            output_node_names=','.join([name.split(":")[0] for name in out_tensor_names]),
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=output_path,
            clear_devices=True,
            initializer_nodes="",
            variable_names_whitelist="",
            variable_names_blacklist="",
            input_meta_graph=None,
            input_saved_model_dir=export_dir,
            saved_model_tags=','.join(DEFAULT_TAGS),
            checkpoint_version=2,
        )

        message_input = "input node name: {in_tensor_names}".format(in_tensor_names=in_tensor_names)
        message_output = "output node name: {out_tensor_names}".format(out_tensor_names=out_tensor_names)

        maxlen = max(len(message_input), len(message_output))

        print("#" * maxlen)
        print(message_input)
        print(message_output)
        print("#" * maxlen)

        print("### Freezing saved model in {export_dir} into {output_path} ###".format(export_dir=export_dir, output_path=output_path))