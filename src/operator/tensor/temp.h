#ifndef MXNET_OPERATOR_TENSOR_TEMP_H_
#define MXNET_OPERATOR_TENSOR_TEMP_H_

#include <mxnet/operator_util.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_broadcast_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct TempAxesParam : public dmlc::Parameter<TempAxesParam> {
  mxnet::TShape axis;
  mxnet::TShape size;
  DMLC_DECLARE_PARAMETER(TempAxesParam) {
    DMLC_DECLARE_FIELD(axis).set_default(mxnet::TShape(0, -1))
      .describe("The axes to perform the broadcasting.");
    DMLC_DECLARE_FIELD(size).set_default(mxnet::TShape(0, -1))
      .describe("Target sizes of the broadcasting axes.");
  }
};

inline bool TempAxesShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector *in_attrs,
                               mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known((*in_attrs)[0])) return false;
  const TempAxesParam& param = nnvm::get<TempAxesParam>(attrs.parsed);
  CHECK_EQ(param.axis.ndim() , param.size.ndim());
  mxnet::TShape &ishape = (*in_attrs)[0];
  mxnet::TShape oshape = ishape;
  for (int i = 0; i < param.axis.ndim(); ++i) {
    CHECK_EQ(oshape[param.axis[i]], 1U) << "Broadcasting axis must have size 1";
    oshape[param.axis[i]] = param.size[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return true;
}

inline void TempReduceShapeCompact(const mxnet::TShape& big, const mxnet::TShape& small,
                                        mxnet::TShape *new_big, mxnet::TShape *new_small) {
  const int idim = std::max(big.ndim(), MXNET_SPECIAL_MAX_NDIM);
  *new_big = mxnet::TShape(idim, 1);
  *new_small = mxnet::TShape(idim, 1);
  index_t j = 0;
  if (small.Size() == 1) {
    (*new_big)[j++] = big.Size();
  } else {
    index_t bprod = 1, sprod = 1;
    for (index_t i = 0, k = 0; i < big.ndim(); ++i) {
      bool red_axis = big[i] != small[i];
      if ((red_axis && sprod > 1) || (!red_axis && bprod != sprod)) {
        (*new_big)[j] = bprod;
        (*new_small)[j] = sprod;
        bprod = sprod = 1; ++j;
      }
      bprod *= big[i];
      if (red_axis) {
        ++k;
      } else {
        sprod *= big[i];
      }
    }
    if (bprod > 1 || sprod > 1) {
      (*new_big)[j] = bprod;
      (*new_small)[j] = sprod;
      ++j;
    }
  }
  if (j <= MXNET_SPECIAL_MAX_NDIM) {
    const int ndim = (j <= 2? 2 : MXNET_SPECIAL_MAX_NDIM);
    new_small->assign(new_small->begin(), new_small->begin() + ndim);
    new_big->assign(new_big->begin(), new_big->begin() + ndim);
  } else {
    LOG(FATAL) << "Too many reduction axes from " << big << " to " << small;
  }
}

namespace {  // unnamed namespace to keep scope of the struct within the file
struct ShapeAndStride {
  index_t in_stride[MXNET_SPECIAL_MAX_NDIM];
  index_t out_stride[MXNET_SPECIAL_MAX_NDIM];
  index_t input_shape[MXNET_SPECIAL_MAX_NDIM];
  index_t output_shape[MXNET_SPECIAL_MAX_NDIM];
  // axes: stores which axes in input is to broadcasted
  index_t axes[MXNET_SPECIAL_MAX_NDIM];
  int num_broadcast_axes;
  bool shape_changed;
};
}
/*!
 * \brief Calculates Stride of input and output tensor dimesnions
          And saves mshadow::Shape data in an integer array for
          faster access.
 * \param *aux_data to hold stride and shape data.
 * \param in_shape input shape
 * \param out_shape output shape
 * \param ndim no of dimensions in output
 */
inline void PrepareAUXData(ShapeAndStride *aux_data,
                    mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape,
                    mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape,
                    int ndim) {
  aux_data->shape_changed = false;
  int iter = ndim - 1, i = 0;
  aux_data->out_stride[iter] = 1;
  aux_data->in_stride[iter] = 1;
  aux_data->input_shape[iter] = in_shape[iter];
  aux_data->output_shape[iter] = out_shape[iter];
  if(in_shape[iter] != out_shape[iter]){
    aux_data->shape_changed = true;
    aux_data->axes[i++] = iter;
  }
  iter--;
  for (; iter >= 0; --iter) {
    aux_data->out_stride[iter] = aux_data->out_stride[iter + 1] * out_shape[iter + 1];
    aux_data->in_stride[iter] = aux_data->in_stride[iter + 1] * in_shape[iter + 1];
    aux_data->input_shape[iter] = in_shape[iter];
    aux_data->output_shape[iter] = out_shape[iter];
    if(in_shape[iter] != out_shape[iter]){
      aux_data->shape_changed = true;
      aux_data->axes[i++] = iter;
    }
  }
  aux_data->num_broadcast_axes = i;
}
//}  // unnamed namespace

template<typename OP>
struct temp_kernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  IType *input,
                                  OType *output,
                                  const ShapeAndStride& aux_data,
                                  const OpReqType req,
                                  const int ndim) {
    index_t idx = i;
    index_t in_idx = i;
#pragma unroll 4
    for (int iter = ndim - 1; iter >= 0; --iter) {
      index_t out_dim_shape = aux_data.output_shape[iter];
      index_t out_dim_stride = aux_data.out_stride[iter];
      // x % y = x - (x / y) * y
      // speeds up modulo(%) operation in GPU
      index_t dim_idx = idx - (idx / out_dim_shape) * out_dim_shape;
      if (aux_data.input_shape[iter] != 1) {
        in_idx += dim_idx * (aux_data.in_stride[iter] - out_dim_stride);
      } else {
        in_idx -= dim_idx * out_dim_stride;
      }
      idx /= out_dim_shape;
    }
    KERNEL_ASSIGN(output[i], req, OP::Map(input[in_idx]));
  }
};

template<typename OP, typename IType, typename OType>
MSHADOW_XINLINE static void b_axis(index_t i,
                                IType *input,
                                OType *output,
                                const ShapeAndStride& aux_data,
                                const OpReqType req,
                                const uint32_t ndim) {
  index_t idx = i;
  index_t init_off = 0;
  for (int iter = ndim - 1; idx > 0 && iter >= 0; --iter) {
    size_t dim_idx = idx % aux_data.input_shape[iter];
    init_off += dim_idx * aux_data.out_stride[iter];
    idx /= aux_data.input_shape[iter];
  }
  index_t stride_0, stride_1, stride_2;
  IType val = input[i];
  // Each case is based on the number of axis to be broadcasted
  // (1, 2 or 3) after merging axes.
  switch (aux_data.num_broadcast_axes) {
    // when input shape is amogst one of the form
    // [(x,1), (x,1,x), (1,x)]
    // x can be any +ve number >=0 and they need not be equal to each other
    case 1 :
      stride_0 = aux_data.out_stride[aux_data.axes[0]];
#pragma omp parallel for
      for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
        //if(omp_get_thread_num() == 0)
        //  printf("Work1 started by tid %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
        KERNEL_ASSIGN(output[init_off + l*stride_0],
            req, OP::Map(val));
      }
      break;
    // when input shape is amogst one of the form
    // [(x,1,x,1), (1,x,1,x), (x,1,x,1,x)]
    // x can be any +ve number >1 or =0(the axis ) and they need not be equal to each other
    case 2:
      stride_1 = aux_data.out_stride[aux_data.axes[1]];
      stride_0 = aux_data.out_stride[aux_data.axes[0]];
      for (int k=0; k < aux_data.output_shape[aux_data.axes[1]]; k++) {
        for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
          KERNEL_ASSIGN(output[init_off + k*stride_1 + l*stride_0],
              req, OP::Map(input[i]));
        }
      }
      break;
    // when input shape is of the form [(1,x,1,x,1)] and
    // x can be any +ve number >=0 and they need not be equal to each other
    case 3:
      stride_2 = aux_data.out_stride[aux_data.axes[2]];
      stride_1 = aux_data.out_stride[aux_data.axes[1]];
      stride_0 = aux_data.out_stride[aux_data.axes[0]];
      for (int j=0; j < aux_data.output_shape[aux_data.axes[2]]; j++) {
        for (int k=0; k < aux_data.output_shape[aux_data.axes[1]]; k++) {
          for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
            KERNEL_ASSIGN(output[init_off + j*stride_2 + k*stride_1 + l*stride_0],
                req, OP::Map(input[i]));
          }
        }
      }
      break;
  }
}

/**
 * Changed the thread workload mapping from 1
 * thread/output element to 1 thread/input to be broadcasted
 * This approach leverages vectorization when fastest varying
 * index(stride=1) of the tensor is to be broadcasted.
 * In other cases it simply performs better by better load balancing.
 */
template<typename OP>
struct temp_kernel_cpu {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  IType *input,
                                  OType *output,
                                  const ShapeAndStride& aux_data,
                                  const OpReqType req,
                                  const uint32_t ndim) {
    index_t idx = i;
    index_t init_off = 0;
    for (int iter = ndim - 1; idx > 0 && iter >= 0; --iter) {
      size_t dim_idx = idx % aux_data.input_shape[iter];
      init_off += dim_idx * aux_data.out_stride[iter];
      idx /= aux_data.input_shape[iter];
    }
    index_t stride_0, stride_1, stride_2;
    // Each case is based on the number of axis to be broadcasted
    // (1, 2 or 3) after merging axes.
    switch (aux_data.num_broadcast_axes) {
      // when input shape is amogst one of the form
      // [(x,1), (x,1,x), (1,x)]
      // x can be any +ve number >=0 and they need not be equal to each other
      case 1 :
        stride_0 = aux_data.out_stride[aux_data.axes[0]];
//#pragma omp parallel for
        for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
          //if(omp_get_thread_num() == 0)
          //  printf("Work1 started by tid %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
          //KERNEL_ASSIGN(output[init_off + l*stride_0],
          //    req, OP::Map(input[i]));
          KERNEL_ASSIGN(output[init_off + l*stride_0],
              req, OP::Map(input[i]));
        }
        break;   //dqefwe
      // when input shape is amogst one of the form
      // [(x,1,x,1), (1,x,1,x), (x,1,x,1,x)]
      // x can be any +ve number >1 or =0(the axis ) and they need not be equal to each other
      case 2:
        stride_1 = aux_data.out_stride[aux_data.axes[1]];
  stride_0 = aux_data.out_stride[aux_data.axes[0]];
        for (int k=0; k < aux_data.output_shape[aux_data.axes[1]]; k++) {
          for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
            KERNEL_ASSIGN(output[init_off + k*stride_1 + l*stride_0],
                req, OP::Map(input[i]));
          }
        }
        break;
      // when input shape is of the form [(1,x,1,x,1)] and
      // x can be any +ve number >=0 and they need not be equal to each other
      case 3:
        stride_2 = aux_data.out_stride[aux_data.axes[2]];
  stride_1 = aux_data.out_stride[aux_data.axes[1]];
        stride_0 = aux_data.out_stride[aux_data.axes[0]];
        for (int j=0; j < aux_data.output_shape[aux_data.axes[2]]; j++) {
          for (int k=0; k < aux_data.output_shape[aux_data.axes[1]]; k++) {
            for (int l=0; l < aux_data.output_shape[aux_data.axes[0]]; l++) {
              KERNEL_ASSIGN(output[init_off + j*stride_2 + k*stride_1 + l*stride_0],
                  req, OP::Map(input[i]));
            }
          }
        }
        break;
    }
  }
};

template<typename xpu>
inline void TempComputeImpl(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs,
                                 const mxnet::TShape& small) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  mxnet::TShape src_shape, dst_shape;
  // combines 2 or more consecutive broadcast/non-broadcast axes together
  // e.g. (3,4,1,1,5,1,6,7) (2,3,5) (5,10,9) -> (12,1,5,1,42) (1,3) (50, 9)
  //      and this is the new input for broadcast_kernel whose total
  //      num of dimensions cannot be greater than 5(throws an error otherwise).
  TempReduceShapeCompact(outputs[0].shape_, small, &dst_shape, &src_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  bool isCPU = ctx.run_ctx.get_ctx().dev_type == Context::kCPU;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, IType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape;
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape;
      for (int i = 0; i < MXNET_SPECIAL_MAX_NDIM; ++i) {
        if (i < dst_shape.ndim()) {
          in_shape[i] = src_shape[i];
          out_shape[i] = dst_shape[i];
        } else {
          in_shape[i] = 1;
          out_shape[i] = 1;
        }
      }
      struct ShapeAndStride aux_data;
      PrepareAUXData(&aux_data, in_shape, out_shape, dst_shape.ndim());
      if(!aux_data.shape_changed) {
        // If no broadcast is required (i.e. input_shape == output_shape)
        // then simply copy input to outout.
        mxnet_op::copy(ctx.get_stream<xpu>(), outputs[0], inputs[0]);
      } else if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, OType> out =
          outputs[0].get_with_shape<xpu, 2, OType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, IType> data =
          inputs[0].get_with_shape<xpu, 2, IType>(src_shape.get<2>(), s);
        if(isCPU) {
          for(int i=0; i<data.shape_.Size(); i++) {
            b_axis<mshadow_op::identity, IType, OType>(i, data.dptr_, out.dptr_, aux_data, req[0], 2);
          }
        //  Kernel<temp_kernel_cpu<mshadow_op::identity>, xpu>::Launch(
        //    s, data.shape_.Size(), data.dptr_, out.dptr_, aux_data, req[0], 2);
        } else {
          Kernel<temp_kernel<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, aux_data, req[0], 2);
        }
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, OType> out =
          outputs[0].get_with_shape<xpu, ndim, OType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, IType> data =
          inputs[0].get_with_shape<xpu, ndim, IType>(src_shape.get<ndim>(), s);
        if(isCPU) {
          for(int i=0; i<data.shape_.Size(); i++) {
            b_axis<mshadow_op::identity, IType, OType>(i, data.dptr_, out.dptr_, aux_data, req[0], ndim);
          }
        //  Kernel<temp_kernel_cpu<mshadow_op::identity>, xpu>::Launch(
        //    s, data.shape_.Size(), data.dptr_, out.dptr_, aux_data, req[0], ndim);
        } else {
          Kernel<temp_kernel<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, aux_data, req[0], ndim);
        }
      }
    });
  });
}

template<typename xpu>
inline void TempCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  TempComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, inputs[0].shape_);
}

#define MXNET_OPERATOR_REGISTER_BROADCAST(name)                 \
  NNVM_REGISTER_OP(name)                                        \
  .set_num_inputs(1)                                            \
  .set_num_outputs(1)                                           \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>) \
  .set_attr<nnvm::FGradient>("FGradient",                       \
    [](const nnvm::ObjectPtr& n,                                  \
       const std::vector<nnvm::NodeEntry>& ograds) {            \
      return MakeNonlossGradNode("_broadcast_backward", n, ograds, {},    \
                                 {{"keepdims", "true"}});              \
    })                                                          \
  .add_argument("data", "NDArray-or-Symbol", "The input")

}
}
#endif
