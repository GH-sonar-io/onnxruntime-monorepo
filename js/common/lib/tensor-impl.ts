// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { tensorToDataURL, tensorToImageData } from './tensor-conversion-impl.js';
import { TensorToDataUrlOptions, TensorToImageDataOptions } from './tensor-conversion.js';
import {
  tensorFromGpuBuffer,
  tensorFromImage,
  tensorFromMLTensor,
  tensorFromPinnedBuffer,
  tensorFromTexture,
} from './tensor-factory-impl.js';
import {
  CpuPinnedConstructorParameters,
  GpuBufferConstructorParameters,
  MLTensorConstructorParameters,
  TensorFromGpuBufferOptions,
  TensorFromImageBitmapOptions,
  TensorFromImageDataOptions,
  TensorFromImageElementOptions,
  TensorFromMLTensorOptions,
  TensorFromTextureOptions,
  TensorFromUrlOptions,
  TextureConstructorParameters,
} from './tensor-factory.js';
import {
  checkTypedArray,
  NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP,
  NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP,
  SupportedTypedArray,
  SupportedTypedArrayConstructors,
} from './tensor-impl-type-mapping.js';
import { calculateSize, tensorReshape } from './tensor-utils-impl.js';
import { Tensor as TensorInterface } from './tensor.js';

// type aliases for those exported from Tensor interface

type TensorType = TensorInterface.Type;
type TensorDataType = TensorInterface.DataType;
type TensorDataLocation = TensorInterface.DataLocation;
type TensorTextureType = TensorInterface.TextureType;
type TensorGpuBufferType = TensorInterface.GpuBufferType;
type TensorMLTensorType = TensorInterface.MLTensorType;
type CpuTensorDataArg = TensorDataType | Uint8ClampedArray | readonly number[] | readonly string[] | readonly boolean[];
type CpuTensorDataSource = TensorDataType | Uint8ClampedArray | readonly string[] | readonly boolean[];
type LocationConstructorParameters =
  | CpuPinnedConstructorParameters
  | TextureConstructorParameters
  | GpuBufferConstructorParameters
  | MLTensorConstructorParameters;

const GPU_BUFFER_SUPPORTED_TYPES = new Set<TensorType>([
  'float32',
  'float16',
  'int32',
  'int64',
  'uint32',
  'uint8',
  'bool',
  'uint4',
  'int4',
]);

const ML_TENSOR_SUPPORTED_TYPES = new Set<TensorType>([
  'float32',
  'float16',
  'int32',
  'int64',
  'uint32',
  'uint64',
  'int8',
  'uint8',
  'bool',
  'uint4',
  'int4',
]);

const validatePinnedBuffer = (type: TensorType, data: TensorDataType): void => {
  const expectedTypedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(type);
  if (!expectedTypedArrayConstructor) {
    throw new TypeError(`unsupported type "${type}" to create tensor from pinned buffer`);
  }
  if (!(data instanceof expectedTypedArrayConstructor)) {
    throw new TypeError(`buffer should be of type ${expectedTypedArrayConstructor.name}`);
  }
};

const createNumericTensorDataFromArray = (
  type: TensorType,
  typedArrayConstructor: SupportedTypedArrayConstructors,
  arg1: CpuTensorDataArg,
): TensorDataType => {
  if ((type === 'float16' && typedArrayConstructor === Uint16Array) || type === 'uint4' || type === 'int4') {
    // - 'float16':
    //   When no Float16Array polyfill is used, we cannot create 'float16' tensor from number array.
    //
    //   Throw error here because when user try to use number array as data,
    //   e.g. new Tensor('float16', [1, 2, 3, 4], dims)), it will actually call
    //   Uint16Array.from(arg1) which generates wrong data.
    //
    // - 'uint4' and 'int4':
    //   Uint8Array.from(arg1) will generate wrong data for 'uint4' and 'int4' tensor.
    //
    throw new TypeError(
      `Creating a ${type} tensor from number array is not supported. Please use ${typedArrayConstructor.name} as data.`,
    );
  }
  if (type === 'uint64' || type === 'int64') {
    // use 'as any' here because:
    // 1. TypeScript's check on type of 'Array.isArray()' does not work with readonly arrays.
    // see https://github.com/microsoft/TypeScript/issues/17002
    // 2. TypeScript's check on union type of '(BigInt64ArrayConstructor|BigUint64ArrayConstructor).from()'
    // does not accept parameter mapFn.
    // 3. parameters of 'SupportedTypedArrayConstructors.from()' does not match the requirement of the union
    // type.

    // assume 'arg1' is of type "readonly number[]|readonly bigint[]" here.

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (typedArrayConstructor as any).from(arg1, BigInt);
  }
  // assume 'arg1' is of type "readonly number[]" here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (typedArrayConstructor as any).from(arg1);
};

const createNumericTensorData = (type: TensorType, arg1: CpuTensorDataArg | undefined): TensorDataType => {
  const typedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(type);
  if (typedArrayConstructor === undefined) {
    throw new TypeError(`Unsupported tensor type: ${type}.`);
  }
  if (Array.isArray(arg1)) {
    return createNumericTensorDataFromArray(type, typedArrayConstructor, arg1);
  }
  if (arg1 instanceof typedArrayConstructor) {
    return arg1;
  }
  if (arg1 instanceof Uint8ClampedArray) {
    if (type === 'uint8') {
      return Uint8Array.from(arg1);
    }
    throw new TypeError(`A Uint8ClampedArray tensor's data must be type of uint8`);
  }
  if (type === 'float16' && arg1 instanceof Uint16Array && typedArrayConstructor !== Uint16Array) {
    // when Float16Array is available and data is of type Uint16Array.
    // We allow Uint16Array to be passed in as data for 'float16' tensor until Float16Array is generally
    // supported in JavaScript environment.

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return new (globalThis as any).Float16Array(arg1.buffer, arg1.byteOffset, arg1.length);
  }
  throw new TypeError(`A ${type} tensor's data must be type of ${typedArrayConstructor}`);
};

/**
 * create tensor data from the given type and data. Used by override: constructor(type, data, ...)
 */
const createTensorDataFromType = (type: TensorType, arg1: CpuTensorDataArg | undefined): TensorDataType => {
  if (type === 'string') {
    if (!Array.isArray(arg1)) {
      throw new TypeError("A string tensor's data must be a string array.");
    }
    // we don't check whether every element in the array is string; this is too slow. we assume it's correct and
    // error will be populated at inference
    return arg1;
  }
  return createNumericTensorData(type, arg1);
};

/**
 * infer tensor type and data from the given data. Used by override: constructor(data, ...)
 */
const inferTypeAndData = (arg0: CpuTensorDataSource): { type: TensorType; data: TensorDataType } => {
  if (Array.isArray(arg0)) {
    // only boolean[] and string[] is supported
    if (arg0.length === 0) {
      throw new TypeError('Tensor type cannot be inferred from an empty array.');
    }
    const firstElementType = typeof arg0[0];
    if (firstElementType === 'string') {
      return { type: 'string', data: arg0 };
    }
    if (firstElementType === 'boolean') {
      // 'arg0' is of type 'boolean[]'. Uint8Array.from(boolean[]) actually works, but typescript thinks this is
      // wrong type. We use 'as any' to make it happy.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return { type: 'bool', data: Uint8Array.from(arg0 as any[]) };
    }
    throw new TypeError(`Invalid element type of data array: ${firstElementType}.`);
  }
  if (arg0 instanceof Uint8ClampedArray) {
    return { type: 'uint8', data: Uint8Array.from(arg0) };
  }
  // get tensor type from TypedArray
  const mappedType = NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.get(arg0.constructor as SupportedTypedArrayConstructors);
  if (mappedType === undefined) {
    throw new TypeError(`Unsupported type for tensor data: ${arg0.constructor}.`);
  }
  return { type: mappedType, data: arg0 as SupportedTypedArray };
};

/**
 * parse the arguments for constructing a tensor of location 'cpu'.
 */
const parseCpuTensorArgs = (
  arg0: TensorType | CpuTensorDataSource,
  arg1?: CpuTensorDataArg,
  arg2?: readonly number[],
): { type: TensorType; data: TensorDataType; dims: readonly number[] } => {
  let type: TensorType;
  let data: TensorDataType;
  let maybeDims: typeof arg1 | typeof arg2;
  if (typeof arg0 === 'string') {
    // Override: constructor(type, data, ...)
    type = arg0;
    maybeDims = arg2;
    data = createTensorDataFromType(arg0, arg1);
  } else {
    // Override: constructor(data, ...)
    maybeDims = arg1;
    ({ type, data } = inferTypeAndData(arg0));
  }

  if (maybeDims === undefined) {
    // assume 1-D tensor if dims omitted
    maybeDims = [data.length];
  } else if (!Array.isArray(maybeDims)) {
    throw new TypeError("A tensor's dims must be a number array");
  }
  return { type, data, dims: maybeDims as readonly number[] };
};

/**
 * check whether the CPU data length matches the tensor size.
 */
const validateDataLength = (type: TensorType, size: number, cpuData: TensorDataType | undefined): void => {
  if (!cpuData || size === cpuData.length) {
    return;
  }
  // for (u)int4, the data length is half of the tensor size. So we check this special case when size is odd.
  if ((type === 'uint4' || type === 'int4') && Math.ceil(size / 2) === cpuData.length) {
    return;
  }
  throw new Error(`Tensor's size(${size}) does not match data length(${cpuData.length}).`);
};

/**
 * the implementation of Tensor interface.
 *
 * @ignore
 */
export class Tensor implements TensorInterface {
  // #region constructors

  /**
   * Construct a new CPU tensor object from the given type, data and dims.
   */
  constructor(
    type: TensorType,
    data: TensorDataType | Uint8ClampedArray | readonly string[] | readonly number[] | readonly boolean[],
    dims?: readonly number[],
  );
  /**
   * Construct a new CPU tensor object from the given data and dims. Type is inferred from data.
   */
  constructor(
    data: TensorDataType | Uint8ClampedArray | readonly string[] | readonly boolean[],
    dims?: readonly number[],
  );
  /**
   * Construct a new tensor object from the pinned CPU data with the given type and dims.
   *
   * Tensor's location will be set to 'cpu-pinned'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: CpuPinnedConstructorParameters);
  /**
   * Construct a new tensor object from the WebGL texture with the given type and dims.
   *
   * Tensor's location will be set to 'texture'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: TextureConstructorParameters);
  /**
   * Construct a new tensor object from the WebGPU buffer with the given type and dims.
   *
   * Tensor's location will be set to 'gpu-buffer'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: GpuBufferConstructorParameters);

  /**
   * Construct a new tensor object from the WebNN MLTensor with the given type and dims.
   *
   * Tensor's location will be set to 'ml-tensor'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: MLTensorConstructorParameters);

  /**
   * implementation.
   */
  constructor(
    arg0:
      | TensorType
      | TensorDataType
      | Uint8ClampedArray
      | readonly string[]
      | readonly boolean[]
      | CpuPinnedConstructorParameters
      | TextureConstructorParameters
      | GpuBufferConstructorParameters
      | MLTensorConstructorParameters,
    arg1?: TensorDataType | Uint8ClampedArray | readonly number[] | readonly string[] | readonly boolean[],
    arg2?: readonly number[],
  ) {
    // perform one-time check for BigInt/Float16Array support
    checkTypedArray();

    let type: TensorType;
    let dims: readonly number[];

    if (typeof arg0 === 'object' && 'location' in arg0) {
      //
      // constructing tensor from specific location
      //
      this.dataLocation = arg0.location;
      type = arg0.type;
      dims = arg0.dims;
      this.initFromLocation(arg0);
    } else {
      //
      // constructing tensor of location 'cpu'
      //
      const parsed = parseCpuTensorArgs(arg0 as TensorType | CpuTensorDataSource, arg1, arg2);
      type = parsed.type;
      dims = parsed.dims;
      this.cpuData = parsed.data;
      this.dataLocation = 'cpu';
    }

    // perform check on dims
    const size = calculateSize(dims);
    // if data is on CPU, check whether data length matches tensor size
    validateDataLength(type, size, this.cpuData);

    this.type = type;
    this.dims = dims;
    this.size = size;
  }

  /**
   * initialize the tensor data fields from the given location-specific constructor parameters.
   */
  private initFromLocation(params: LocationConstructorParameters): void {
    const type = params.type;
    switch (params.location) {
      case 'cpu-pinned': {
        validatePinnedBuffer(type, params.data);
        this.cpuData = params.data;
        break;
      }
      case 'texture': {
        if (type !== 'float32') {
          throw new TypeError(`unsupported type "${type}" to create tensor from texture`);
        }
        this.gpuTextureData = params.texture;
        this.downloader = params.download;
        this.disposer = params.dispose;
        break;
      }
      case 'gpu-buffer': {
        if (!GPU_BUFFER_SUPPORTED_TYPES.has(type)) {
          throw new TypeError(`unsupported type "${type}" to create tensor from gpu buffer`);
        }
        this.gpuBufferData = params.gpuBuffer;
        this.downloader = params.download;
        this.disposer = params.dispose;
        break;
      }
      case 'ml-tensor': {
        if (!ML_TENSOR_SUPPORTED_TYPES.has(type)) {
          throw new TypeError(`unsupported type "${type}" to create tensor from MLTensor`);
        }
        this.mlTensorData = params.mlTensor;
        this.downloader = params.download;
        this.disposer = params.dispose;
        break;
      }
      default:
        throw new Error(`Tensor constructor: unsupported location '${this.dataLocation}'`);
    }
  }
  // #endregion

  // #region factory
  static async fromImage(
    image: ImageData | HTMLImageElement | ImageBitmap | string,
    options?:
      | TensorFromImageDataOptions
      | TensorFromImageElementOptions
      | TensorFromImageBitmapOptions
      | TensorFromUrlOptions,
  ): Promise<TensorInterface> {
    return tensorFromImage(image, options);
  }

  static fromTexture<T extends TensorInterface.TextureDataTypes>(
    texture: TensorTextureType,
    options: TensorFromTextureOptions<T>,
  ): TensorInterface {
    return tensorFromTexture(texture, options);
  }

  static fromGpuBuffer<T extends TensorInterface.GpuBufferDataTypes>(
    gpuBuffer: TensorGpuBufferType,
    options: TensorFromGpuBufferOptions<T>,
  ): TensorInterface {
    return tensorFromGpuBuffer(gpuBuffer, options);
  }

  static fromMLTensor<T extends TensorInterface.MLTensorDataTypes>(
    mlTensor: TensorMLTensorType,
    options: TensorFromMLTensorOptions<T>,
  ): TensorInterface {
    return tensorFromMLTensor(mlTensor, options);
  }

  static fromPinnedBuffer<T extends TensorInterface.CpuPinnedDataTypes>(
    type: T,
    buffer: TensorInterface.DataTypeMap[T],
    dims?: readonly number[],
  ): Tensor {
    return tensorFromPinnedBuffer(type, buffer, dims);
  }

  // #endregion

  // #region conversions
  toDataURL(options?: TensorToDataUrlOptions): string {
    return tensorToDataURL(this, options);
  }

  toImageData(options?: TensorToImageDataOptions): ImageData {
    return tensorToImageData(this, options);
  }
  // #endregion

  // #region public fields
  readonly dims: readonly number[];
  readonly type: TensorType;
  readonly size: number;
  // #endregion

  // #region private fields

  /**
   * stores the location of the data.
   */
  private dataLocation: TensorDataLocation;

  /**
   * stores the data on CPU, if location is 'cpu' or 'cpu-pinned'. otherwise empty.
   */
  private cpuData?: TensorDataType;

  /**
   * stores the underlying texture when location is 'texture'. otherwise empty.
   */
  private gpuTextureData?: TensorTextureType;

  /**
   * stores the underlying GPU buffer when location is 'gpu-buffer'. otherwise empty.
   */
  private gpuBufferData?: TensorGpuBufferType;

  /**
   * stores the underlying WebNN MLTensor when location is 'ml-tensor'. otherwise empty.
   */
  private mlTensorData?: TensorMLTensorType;

  /**
   * stores an optional downloader function to download data from GPU to CPU.
   */
  private downloader?(): Promise<TensorDataType>;

  /**
   * a flag indicating whether the data is being downloaded from GPU to CPU.
   */
  private isDownloading?: boolean;

  /**
   * stores an optional disposer function to dispose the underlying data.
   */
  private disposer?(): void;
  // #endregion

  // #region properties
  get data(): TensorDataType {
    this.ensureValid();
    if (!this.cpuData) {
      throw new Error(
        'The data is not on CPU. Use `getData()` to download GPU data to CPU, ' +
          'or use `texture` or `gpuBuffer` property to access the GPU data directly.',
      );
    }
    return this.cpuData;
  }

  get location(): TensorDataLocation {
    return this.dataLocation;
  }

  get texture(): TensorTextureType {
    this.ensureValid();
    if (!this.gpuTextureData) {
      throw new Error('The data is not stored as a WebGL texture.');
    }
    return this.gpuTextureData;
  }

  get gpuBuffer(): TensorGpuBufferType {
    this.ensureValid();
    if (!this.gpuBufferData) {
      throw new Error('The data is not stored as a WebGPU buffer.');
    }
    return this.gpuBufferData;
  }

  get mlTensor(): TensorMLTensorType {
    this.ensureValid();
    if (!this.mlTensorData) {
      throw new Error('The data is not stored as a WebNN MLTensor.');
    }
    return this.mlTensorData;
  }
  // #endregion

  // #region methods

  async getData(releaseData?: boolean): Promise<TensorDataType> {
    this.ensureValid();
    switch (this.dataLocation) {
      case 'cpu':
      case 'cpu-pinned':
        return this.data;
      case 'texture':
      case 'gpu-buffer':
      case 'ml-tensor': {
        if (!this.downloader) {
          throw new Error('The current tensor is not created with a specified data downloader.');
        }
        if (this.isDownloading) {
          throw new Error('The current tensor is being downloaded.');
        }
        try {
          this.isDownloading = true;
          const data = await this.downloader();
          this.downloader = undefined;
          this.dataLocation = 'cpu';
          this.cpuData = data;

          if (releaseData && this.disposer) {
            this.disposer();
            this.disposer = undefined;
          }

          return data;
        } finally {
          this.isDownloading = false;
        }
      }
      default:
        throw new Error(`cannot get data from location: ${this.dataLocation}`);
    }
  }

  dispose(): void {
    if (this.isDownloading) {
      throw new Error('The current tensor is being downloaded.');
    }

    if (this.disposer) {
      this.disposer();
      this.disposer = undefined;
    }
    this.cpuData = undefined;
    this.gpuTextureData = undefined;
    this.gpuBufferData = undefined;
    this.mlTensorData = undefined;
    this.downloader = undefined;
    this.isDownloading = undefined;

    this.dataLocation = 'none';
  }

  // #endregion

  // #region tensor utilities
  private ensureValid(): void {
    if (this.dataLocation === 'none') {
      throw new Error('The tensor is disposed.');
    }
  }

  reshape(dims: readonly number[]): TensorInterface {
    this.ensureValid();
    if (this.downloader || this.disposer) {
      throw new Error('Cannot reshape a tensor that owns GPU resource.');
    }
    return tensorReshape(this, dims);
  }
  // #endregion
}
