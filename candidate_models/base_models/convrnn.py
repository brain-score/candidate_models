import tensorflow as tf
from tnn import main as tnn_main
from tnn.reciprocalgaternn import tnn_ReciprocalGateCell

dropout10L = {'conv'+str(l):1.0 for l in range(1,11)}
dropout10L['imnetds'] = 1.0
#todo need to fill in with the npz params
def tnn_base_edges(inputs, train=True, basenet_layers=['conv'+str(l) for l in range(1,11)], alter_layers=None,
             unroll_tf=False, const_pres=False, out_layers='imnetds', base_name='model_jsons/10Lv9_imnet128_res23_rrgctx', 
             times=range(18), image_on=0, image_off=11, delay=10, random_off=None, dropout=dropout10L, 
             edges_arr=[], convrnn_type='recipcell', mem_val=0.0, train_tau_fg=False, apply_bn=False,
             channel_op='concat', seed=0, min_duration=11, 
             layer_params={},
             p_edge=1.0,
             decoder_start=18,
             decoder_end=26,
             decoder_type='last',
             ff_weight_decay=0.0,
             ff_kernel_initializer_kwargs={},
             final_max_pool=True,
             tpu_name=None,
             gcp_project=None,
             tpu_zone=None,
             num_shards=None,
             iterations_per_loop=None, **kwargs):  

    mo_params = {}
    print("using multicell model!")
    # set ds dropout
    # dropout[out_layers] = ds_dropout
    
    # times may be a list or array, where t = 10t-10(t+1)ms.
    # if times is a list, it must be a subset of range(26).
    # input reaches convT at time t (no activations at t=0)
    image_off = int(image_off)
    decoder_start = int(decoder_start)
    decoder_end = int(decoder_end)
    if isinstance(times, (int, float)):
        ntimes = times
        times = range(ntimes)
    else:
        ntimes = times[-1]+1
    
    if random_off is not None and train == True:
        print("max duration", random_off - image_on)
        print("min duration", min_duration)
        image_times = np.random.choice(range(min_duration, random_off - image_on + 1))
        image_off = image_on + image_times
        print("image times", image_times)
        times = range(image_on + delay, image_off + delay)
        readout_time = times[-1]
        print("model times", times)
        print("readout_time", readout_time)
    else:
        image_times = image_off - image_on
    
    # set up image presentation, note that inputs is a tensor now, not a dictionary
    ims = tf.identity(tf.cast(inputs, dtype=tf.float32), name='split')
    batch_size = ims.get_shape().as_list()[0]
    print('IM SHAPE', ims.shape)
    
    if const_pres:
        print('Using constant image presentation')
        pres = ims
    else:
        print('Making movie')
        blank = tf.constant(value=0.5, shape=ims.get_shape().as_list(), name='split')
        pres = ([blank] * image_on) +  ([ims] * image_times) + ([blank] * (ntimes - image_off))
        
    # graph building stage
    with tf.variable_scope('tnn_model'):
        base_name += '.json'
        print('Using base: ', base_name)
        G = tnn_main.graph_from_json(base_name)
        print("graph build from JSON")
        
        # memory_cell_params = cell_params.copy()
        # print("CELL PARAMS:", cell_params)
            
        # dealing with dropout between training and validation
        for node, attr in G.nodes(data=True):
            if apply_bn:
                if 'conv' in node:
                    print('Applying batch norm to ', node)
                    # set train flag of batch norm for conv layers
                    attr['kwargs']['pre_memory'][0][1]['batch_norm'] = True
                    attr['kwargs']['pre_memory'][0][1]['is_training'] = train

            this_layer_params = layer_params[node]
            # set ksize, out depth, and training flag for batch_norm
            for func, kwargs in attr['kwargs']['pre_memory'] + attr['kwargs']['post_memory']:

                if func.__name__ in ['component_conv', 'conv']:
                    ksize_val = this_layer_params.get('ksize')
                    if ksize_val is not None:
                        kwargs['ksize'] = ksize_val
                    print("using ksize {} for {}".format(kwargs['ksize'], node))
                    out_depth_val = this_layer_params.get('out_depth')
                    if out_depth_val is not None:
                        kwargs['out_depth'] = out_depth_val
                    print("using out depth {} for {}".format(kwargs['out_depth'], node)) 
                    if ff_weight_decay is not None:     # otherwise uses json               
                        kwargs['weight_decay'] = ff_weight_decay
                    if kwargs['kernel_init'] == "variance_scaling":
                        if ff_kernel_initializer_kwargs is not None: # otherwise uses json
                            kwargs['kernel_init_kwargs'] = ff_kernel_initializer_kwargs

                # if func.__name__ == 'dropout':
                #     kwargs['keep_prob'] = dropout[node]

            # # optional max pooling at end of conv10
            if node == 'conv10':
                if final_max_pool:
                    attr['kwargs']['post_memory'][-1] = (tf.nn.max_pool,
                                                        {'ksize': [1,2,2,1],
                                                         'strides': [1,2,2,1],
                                                         'padding': 'SAME'})
                    print("using a final max pool")
                else:
                    attr['kwargs']['post_memory'][-1] = (tf.identity, {})
                    print("not using a final max pool")

            # # optional final average pooling before dropout and fc
            # if node == 'imnetds':
            #     if final_avg_pool:
            #         attr['kwargs']['pre_memory'][0] = (global_pool,
            #                                            {'kind': 'avg'})
            #     else:
            #         attr['kwargs']['pre_memory'][0] = (tf.identity, {})

            # set memory params, including cell config
            memory_func, memory_param = attr['kwargs']['memory']
            
            if any(s in memory_param for s in ('gate_filter_size', 'tau_filter_size')):
                if convrnn_type == 'recipcell':
                    print('using reciprocal gated cell for ', node)
                    attr['cell'] = tnn_ReciprocalGateCell
                    recip_cell_params = this_layer_params['cell_params'].copy()
                    assert recip_cell_params is not None
                    for k,v in recip_cell_params.items():
                        attr['kwargs']['memory'][1][k] = v

            else:
                if alter_layers is None:
                    alter_layers = basenet_layers
                if node in alter_layers: 
                    attr['kwargs']['memory'][1]['memory_decay'] = mem_val
                    attr['kwargs']['memory'][1]['trainable'] = train_tau_fg
                if node in basenet_layers:
                    print(node, attr['kwargs']['memory'][1])

        # add non feedforward edges
        if len(edges_arr) > 0:
            edges = []
            for edge, p in edges_arr:
                if p <= p_edge:
                    edges.append(edge)
            print("applying edges,", edges)
            G.add_edges_from(edges)

        # initialize graph structure
        tnn_main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        
        # unroll graph
        if unroll_tf:
            print('Unroll tf way')
            tnn_main.unroll_tf(G, input_seq={'conv1': pres}, ntimes=ntimes)
        else:
            print('Unrolling tnn way')
            tnn_main.unroll(G, input_seq={'conv1': pres}, ntimes=ntimes)

        # collect last timestep output
        logits_list = [G.node['imnetds']['outputs'][t] for t in range(decoder_start, decoder_end)]

        print("decoder_type", decoder_type, "from", decoder_start, "to", decoder_end)
        if decoder_type == 'last':
            logits = logits_list[-1]
        elif decoder_type == 'sum':
            logits = tf.add_n(logits_list)
        elif decoder_type == 'avg':
            logits = tf.add_n(logits_list) / len(logits_list)
        elif decoder_type == 'random':
            if train:
                logits = np.random.choice(logits_list)
            elif not train: #eval -- use last timepoint with image on
                t_eval = image_off + delay - 1
                t_eval = t_eval - decoder_start
                logits = logits_list[t_eval]

        logits = tf.squeeze(logits)
        print("logits shape", logits.shape)

    outputs = {}
    outputs['imnet_logits'] = logits  
    outputs['times'] = {} 
    for t in times:
        outputs['times'][t] = tf.squeeze(G.node[out_layers]['outputs'][t])     
    return outputs, mo_params
