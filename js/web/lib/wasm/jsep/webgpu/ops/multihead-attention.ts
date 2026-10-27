// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, GpuDataType, ProgramUniform } from '../types';

import {
  applyAttention,
  AttentionAttrs,
  AttentionMaskType,
  AttentionParameters,
  AttentionQkvFormat,
} from './attention';
import { inputVariable, outputVariable, ShaderHelper, UniformsArrayType } from './common';
import { createTransposeProgramInfo, TransposeAttributes } from './transpose';

const getInput = (inputs: readonly TensorView[], i: number) =>
  inputs.length > i && inputs[i].dims.length > 0 ? inputs[i] : undefined;

const validatePastKeyValue = (
  pastKey: TensorView | undefined,
  pastValue: TensorView | undefined,
  batchSize: number,
  numHeads: number,
  headSize: number,
): number => {
  if (pastKey && pastValue && ShapeUtil.size(pastKey.dims) && ShapeUtil.size(pastValue.dims)) {
    if (pastKey.dims.length !== 4) {
      throw new Error('Input "past_key" is expected to have 4 dimensions');
    }
    if (pastKey.dims[0] !== batchSize || pastKey.dims[1] !== numHeads || pastKey.dims[3] !== headSize) {
      throw new Error('Input "past_key" shape (batch_size, num_heads, past_sequence_length, head_size)');
    }
    if (pastValue.dims[0] !== batchSize || pastValue.dims[1] !== numHeads || pastValue.dims[3] !== headSize) {
      throw new Error('Input "past_value" shape (batch_size, num_heads, past_sequence_length, head_size)');
    }
    if (pastKey.dims[2] !== pastValue.dims[2]) {
      throw new Error('Input "past_key" and "past_value" shall have same dim 2 (past_sequence_length)');
    }
    if (pastValue.dims.length !== 4) {
      throw new Error('Input "past_value" is expected to have 4 dimensions');
    }
    return pastKey.dims[2];
  }
  if ((pastKey && ShapeUtil.size(pastKey.dims)) || (pastValue && ShapeUtil.size(pastValue.dims))) {
    throw new Error('Input "past_key" and "past_value" shall be both present or both absent');
  }
  return 0;
};

const validateKey = (
  query: TensorView,
  key: TensorView,
  value: TensorView | undefined,
  numHeads: number,
  headSize: number,
): { qkvFormat: AttentionQkvFormat; kvSequenceLength: number } => {
  if (query.dims.length !== 3) {
    throw new Error('Input "query" is expected to have 3 dimensions when key is given');
  }
  if (key.dims.length < 3 || key.dims.length > 5) {
    throw new Error('Input "key" is expected to have 3, 4, or 5 dimensions');
  }
  if (query.dims[0] !== key.dims[0]) {
    throw new Error('Input "query" and "key" shall have same dim 0 (batch size)');
  }

  if (key.dims.length === 3) {
    if (key.dims[2] !== query.dims[2]) {
      throw new Error('Input "query" and "key" shall have same dim 2 (hidden_size)');
    }
    return { qkvFormat: AttentionQkvFormat.qkvBSNH, kvSequenceLength: key.dims[1] };
  }
  if (key.dims.length === 5) {
    if (key.dims[2] !== numHeads || key.dims[3] !== 2 || key.dims[4] !== headSize) {
      throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');
    }
    if (value) {
      throw new Error('Expect "value" be none when "key" has packed kv format.');
    }
    return { qkvFormat: AttentionQkvFormat.qKvBSNHxBSN2H, kvSequenceLength: key.dims[1] };
  }
  // key_dims.size() == 4 (cross-attention with past_key)
  if (key.dims[1] !== numHeads || key.dims[3] !== headSize) {
    throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');
  }
  // Q_K_V_BSNH_BNSH_BNSH
  return { qkvFormat: AttentionQkvFormat.unknown, kvSequenceLength: key.dims[2] };
};

const validatePackedQkv = (query: TensorView, numHeads: number): void => {
  if (query.dims.length !== 5) {
    throw new Error('Input "query" is expected to have 5 dimensions when key is empty');
  }
  if (query.dims[2] !== numHeads || query.dims[3] !== 3) {
    throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');
  }
};

const validateBias = (bias: TensorView | undefined, key: TensorView | undefined): void => {
  if (!bias || ShapeUtil.size(bias.dims) <= 0) {
    return;
  }
  if (bias.dims.length !== 1) {
    throw new Error('Input "bias" is expected to have 1 dimension');
  }
  if (key?.dims.length === 5 && key.dims[3] === 2) {
    throw new Error('bias is not allowed for packed kv.');
  }
};

const validateKeyPaddingMask = (
  keyPaddingMask: TensorView | undefined,
  batchSize: number,
  totalSequenceLength: number,
): AttentionMaskType => {
  if (!keyPaddingMask || ShapeUtil.size(keyPaddingMask.dims) <= 0) {
    return AttentionMaskType.none;
  }
  let maskType: AttentionMaskType = AttentionMaskType.maskUnknown;
  const maskDims = keyPaddingMask.dims;
  if (maskDims.length === 1) {
    if (maskDims[0] === batchSize) {
      maskType = AttentionMaskType.mask1dKeySeqLen;
    } else if (maskDims[0] === 3 * batchSize + 2) {
      maskType = AttentionMaskType.mask1DKeySeqLenStart;
    }
  } else if (maskDims.length === 2 && maskDims[0] === batchSize && maskDims[1] === totalSequenceLength) {
    maskType = AttentionMaskType.mask2dKeyPadding;
  }
  if (maskType === AttentionMaskType.maskUnknown) {
    throw new Error('Input "key_padding_mask" shape shall be (batch_size) or (batch_size, total_sequence_length)');
  }
  throw new Error('Mask not supported');
};

const validateValue = (
  query: TensorView,
  value: TensorView | undefined,
  kvSequenceLength: number,
  hiddenSize: number,
): { vHiddenSize: number; passPastInKv: boolean } => {
  if (!value || ShapeUtil.size(value.dims) <= 0) {
    return { vHiddenSize: hiddenSize, passPastInKv: false };
  }
  if (value.dims.length !== 3 && value.dims.length !== 4) {
    throw new Error('Input "value" is expected to have 3 or 4 dimensions');
  }
  if (query.dims[0] !== value.dims[0]) {
    throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');
  }
  if (value.dims.length === 3) {
    if (kvSequenceLength !== value.dims[1]) {
      throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');
    }
    return { vHiddenSize: value.dims[2], passPastInKv: false };
  }
  // Q_K_V_BSNH_BNSH_BNSH
  if (kvSequenceLength !== value.dims[2]) {
    throw new Error('Input "key" and "value" shall have the same dim 2 (kv_sequence_length)');
  }
  return { vHiddenSize: value.dims[1] * value.dims[3], passPastInKv: true };
};

const validateAttentionBias = (
  attentionBias: TensorView | undefined,
  batchSize: number,
  numHeads: number,
  sequenceLength: number,
  totalSequenceLength: number,
): void => {
  if (!attentionBias || ShapeUtil.size(attentionBias.dims) <= 0) {
    return;
  }
  if (attentionBias.dims.length !== 4) {
    throw new Error('Input "attention_bias" is expected to have 4 dimensions');
  }

  // TODO: support broadcasting the first and second dimensions of attention_bias.
  if (
    attentionBias.dims[0] !== batchSize ||
    attentionBias.dims[1] !== numHeads ||
    attentionBias.dims[2] !== sequenceLength ||
    attentionBias.dims[3] !== totalSequenceLength
  ) {
    throw new Error('Expect "attention_bias" shape (batch_size, num_heads, sequence_length, total_sequence_length)');
  }
};

const validateInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  const query = inputs[0];
  const key = getInput(inputs, 1);
  const value = getInput(inputs, 2);
  const bias = getInput(inputs, 3);
  const keyPaddingMask = getInput(inputs, 4);
  const attentionBias = getInput(inputs, 5);
  const pastKey = getInput(inputs, 6);
  const pastValue = getInput(inputs, 7);

  // ---------------------------------------------------------------
  // Notations:
  //    B: batch_size
  //    N: num_heads
  //    H: head_size of Q and K
  //    H_v: head_size of V
  //    D: hidden_size for Q and K, where D = N * H
  //    D_v: hidden_size of V, where D_v = N * H_v
  //    S: q_sequence_length
  //    P: past_sequence_length of kv cache
  //    L: kv_sequence_length
  //    T: total_sequence_length = P + L
  //    M: max_sequence_length of kv cache when past and present share buffer
  // ---------------------------------------------------------------
  // MultiHeadAttention inputs:
  // ---------------------------------------------------------------
  //  Q_K_V_BSNH - no packing:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D)
  //     value            (V)       : (B, L, D_v)
  //  Q_K_V_BSNH_BNSH_BNSH - cross attention (kv cache is not used, L == T, D == D_v):
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, N, L, H)
  //     value            (V)       : (B, N, L, H_v)
  //  Q_KV_BSNH_BSN2H - packed kv (kv cache is not used, bias is not allowed for packed kv):
  //     query            (Q)       : (B, S, D)
  //     key              (K/V)     : (B, L, N, 2, H)
  //     value                      : None
  //  QKV_BSN3H - packed qkv (kv cache is not used, S == L, D == D_v):
  //     query            (Q/K/V)   : (B, S, N, 3, H)
  //     key                        : None
  //     value                      : None
  //
  //  Other inputs:
  //     bias             (Q/K/V)   : None or (D + D + D_v)
  //     key_padding_mask (K/V)     : (B) or (3 * B + 2) or (B, T) or (B, S, T)
  //     attention_bias             : None or (B, N, S, T), (1, N, S, T), (B, 1, S, T) or (1, 1, S, T)
  //     past_key                   : (B, N, P, H) or None. Past state is only allowed for Q_K_V_BSNH.
  //     past_value                 : (B, N, P, H) or None. Past state is only allowed for Q_K_V_BSNH.
  //
  //  Not Supported:
  //     key_padding_mask, packed kv, packed qkv, and broadcast for attention_bias.

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = query.dims.length === 3 ? query.dims[2] : attributes.numHeads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  const headSize = Math.floor(hiddenSize / attributes.numHeads);
  const pastSequenceLength = validatePastKeyValue(pastKey, pastValue, batchSize, attributes.numHeads, headSize);
  const maxSequenceLength = pastSequenceLength;

  let qkvFormat: AttentionQkvFormat;
  if (key && ShapeUtil.size(key.dims) > 0) {
    ({ qkvFormat, kvSequenceLength } = validateKey(query, key, value, attributes.numHeads, headSize));
  } else {
    validatePackedQkv(query, attributes.numHeads);
    qkvFormat = AttentionQkvFormat.qkvBSN3H;
  }

  validateBias(bias, key);

  const totalSequenceLength = pastSequenceLength + kvSequenceLength;

  const maskType = validateKeyPaddingMask(keyPaddingMask, batchSize, totalSequenceLength);

  const { vHiddenSize, passPastInKv } = validateValue(query, value, kvSequenceLength, hiddenSize);

  const broadcastResPosBias = false;

  if (keyPaddingMask && ShapeUtil.size(keyPaddingMask.dims) > 0) {
    throw new Error('Key padding mask is not supported');
  }

  validateAttentionBias(attentionBias, batchSize, attributes.numHeads, sequenceLength, totalSequenceLength);

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize,
    vHeadSize: Math.floor(vHiddenSize / attributes.numHeads),
    numHeads: attributes.numHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  };
};

export const parseMultiHeadAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
  createAttributeWithCacheKey({ ...attributes });

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({ perm: [0, 2, 1, 3] });

const addBiasTranspose = (
  context: ComputeContext,
  qkv: TensorView,
  bias: TensorView,
  batchSize: number,
  sequenceLength: number,
  hiddenSize: number,
  biasOffset: number,
) => {
  const outputShape = [batchSize, sequenceLength, hiddenSize];
  const outputSize = ShapeUtil.size(outputShape);
  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: outputSize },
    { type: DataType.uint32, data: biasOffset },
    { type: DataType.uint32, data: hiddenSize },
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const output = outputVariable('qkv_with_bias', qkv.dataType, outputShape);
    const qkvInput = inputVariable('qkv', qkv.dataType, outputShape);
    const biasInput = inputVariable('bias', bias.dataType, outputShape);

    const uniforms: UniformsArrayType = [
      { name: 'output_size', type: 'u32' },
      { name: 'bias_offset', type: 'u32' },
      { name: 'hidden_size', type: 'u32' },
    ];
    return `
  ${shaderHelper.registerUniforms(uniforms).declareVariables(qkvInput, biasInput, output)}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
    let bias_offset_idx = (global_idx % uniforms.hidden_size) + uniforms.bias_offset;

    qkv_with_bias[global_idx] = qkv[global_idx] + bias[bias_offset_idx];
  }`;
  };

  return context.compute(
    {
      name: 'MultiHeadAttentionAddBias',
      shaderCache: { inputDependencies: ['type', 'type'] },
      getRunData: () => ({
        outputs: [{ dims: outputShape, dataType: qkv.dataType, gpuDataType: GpuDataType.default }],
        dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
        programUniforms,
      }),
      getShaderSource,
    },
    { inputs: [qkv, bias], outputs: [-1] },
  )[0];
};

export const maybeTransposeToBNSHAndAddBias = (
  context: ComputeContext,
  batchSize: number,
  numHeads: number,
  sequenceLength: number,
  headSize: number,
  input: TensorView,
  bias?: TensorView,
  biasOffset?: number,
) => {
  // const newDims = [];

  let reshapedInput = input;
  if (!(bias && ShapeUtil.size(bias.dims) > 0)) {
    if (input.dims.length === 3) {
      reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
    }
    if (numHeads === 1 || sequenceLength === 1) {
      return reshapedInput;
    }
    return context.compute(createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm), {
      inputs: [reshapedInput],
      outputs: [-1],
    })[0];
  } else {
    if (sequenceLength === 1) {
      throw new Error('AddBiasReshape is not implemented. Please export your model with packed QKV or KV');
    } else {
      reshapedInput = addBiasTranspose(
        context,
        input,
        bias,
        batchSize,
        sequenceLength,
        numHeads * headSize,
        biasOffset!,
      );
      reshapedInput = reshapedInput.reshape([batchSize, sequenceLength, numHeads, headSize]);
      if (numHeads === 1 || sequenceLength === 1) {
        return reshapedInput;
      }
      return context.compute(createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm), {
        inputs: [reshapedInput],
        outputs: [-1],
      })[0];
    }
  }
};

export const multiHeadAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);
  const query = context.inputs[0];
  const key = getInput(context.inputs, 1);
  const value = getInput(context.inputs, 2);
  const bias = getInput(context.inputs, 3);
  const keyPaddingMask = getInput(context.inputs, 4);
  const attentionBias = getInput(context.inputs, 5);
  const pastKey = getInput(context.inputs, 6);
  const pastValue = getInput(context.inputs, 7);
  if (query.dims.length === 5) {
    throw new Error('Packed QKV is not implemented');
  }

  if (key?.dims.length === 5) {
    throw new Error('Packed KV is not implemented');
  }

  // applyAttention expects BNSH inputs
  const kvBNSH = key && value && key.dims.length === 4 && value.dims.length === 4;

  const Q = maybeTransposeToBNSHAndAddBias(
    context,
    params.batchSize,
    params.numHeads,
    params.sequenceLength,
    params.headSize,
    query,
    bias,
    0,
  );

  if (kvBNSH) {
    return applyAttention(context, Q, key, value, keyPaddingMask, undefined, pastKey, pastValue, attentionBias, params);
  }
  if (!key || !value) {
    throw new Error('key and value must be provided');
  }
  const K = maybeTransposeToBNSHAndAddBias(
    context,
    params.batchSize,
    params.numHeads,
    params.kvSequenceLength,
    params.headSize,
    key,
    bias,
    params.hiddenSize,
  );

  const V = maybeTransposeToBNSHAndAddBias(
    context,
    params.batchSize,
    params.numHeads,
    params.kvSequenceLength,
    params.vHeadSize,
    value,
    bias,
    2 * params.hiddenSize,
  );

  applyAttention(context, Q, K, V, keyPaddingMask, undefined, pastKey, pastValue, attentionBias, params);
};
