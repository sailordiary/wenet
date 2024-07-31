import json
import os
from wenet.experimental.dmels.dmels_asr_model import DmelsAsrModel
from wenet.experimental.dmels.dmels_quantizer import DmelsQuantizer
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import WENET_CTC_CLASSES, WENET_DECODER_CLASSES, WENET_ENCODER_CLASSES


def get_m_M(input_file):
    obj = json.load(input_file)
    return obj['m'], obj['M']


def init_dmels_asr_model(args, configs):
    # TODO(xcsong): Forcefully read the 'cmvn' attribute.
    global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    ctc_type = configs.get('ctc', 'ctc')

    encoder = WENET_ENCODER_CLASSES[encoder_type](
        input_dim,
        global_cmvn=global_cmvn,
        **configs['encoder_conf'],
        **configs['encoder_conf']['efficient_conf']
        if 'efficient_conf' in configs['encoder_conf'] else {})
    decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])

    ctc = WENET_CTC_CLASSES[ctc_type](
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
        if 'ctc_conf' in configs else 0)

    m, M = get_m_M(configs['quantizer']['quantizer_json'])
    quantizer = DmelsQuantizer(m, M)
    model = DmelsAsrModel(quantizer=quantizer,
                          vocab_size=vocab_size,
                          encoder=encoder,
                          decoder=decoder,
                          ctc=ctc,
                          special_tokens=configs.get('tokenizer_conf', {}).get(
                              'special_tokens', None),
                          **configs['model_conf'])

    # If specify checkpoint, load some info from checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    configs["init_infos"] = infos

    if int(os.environ.get('RANK', 0)) == 0:
        print(configs)

    # Tie emb.weight to decoder.output_layer.weight
    if model.decoder.tie_word_embedding:
        if not hasattr(args, 'jit'):
            args.jit = True  # i.e. export onnx/jit/ipex
        model.decoder.tie_or_clone_weights(jit_mode=args.jit)

    return model, configs
