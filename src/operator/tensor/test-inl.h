#ifndef MXNET_OPERATOR_TEST_OP_H_
#define MXNET_OPERATOR_TEST_OP_H_

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

template<typename OP>
struct test_kernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  IType *input,
                                  OType *output,
                                  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape,
                                  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape,
                                  const OpReqType req,
                                  const uint32_t ndim) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t in_idx = i;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % out_shape[iter];
      in_idx -= dim_idx * out_stride;
      if (in_shape[iter] != 1) {
        in_idx += dim_idx * in_stride;
      }
      idx /= out_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    //printf("Thread:%d: getting input from input[%d]\n", i, in_idx);
    KERNEL_ASSIGN(output[i], req, OP::Map(input[in_idx]));
  }
};

template<typename OP>
struct test_kernel_gpu {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int32_t i,
                                  IType *input,
                                  OType *output,
                                  const int32_t *input_shape,
                                  const int32_t *output_shape,
                                  //mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape,
                                  //mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape,
                                  const int32_t *in_stride,
                                  const int32_t *out_stride,
                                  const OpReqType req,
                                  const int32_t ndim) {
    int32_t idx = i;
    int32_t in_idx = i;
    //printf("here1\n");
//    int32_t *input_shape = reinterpret_cast<int32_t*>(in_shape.shape_);
//    int32_t *output_shape = reinterpret_cast<int32_t*>(out_shape.shape_);
    for (int32_t iter = ndim - 1; iter >= 0; --iter) {
//      int32_t in_dim_size = (int32_t)(in_shape.shape_[iter]);
//      printf("in_shape.shape_[%d]=%d and input_shape[%d]=%d\n", iter, (int32_t)in_shape.shape_[iter], iter, (int32_t)input_shape[iter]);
//      int32_t out_dim_size = (int32_t)(out_shape.shape_[iter]);
      int32_t dim_idx = idx % output_shape[iter];
//      int32_t dim_idx = idx - (idx/out_dim_size) * out_dim_size;
      if (input_shape[iter] != 1) {
        in_idx += dim_idx * (in_stride[iter] - out_stride[iter]);
      } else {
        in_idx -= dim_idx * out_stride[iter];
      }
      idx /= output_shape[iter];
    }
    KERNEL_ASSIGN(output[i], req, OP::Map(input[in_idx]));
  }
};

template<int req>
struct comp_offset {
  MSHADOW_XINLINE static void Map(int32_t i,
                                  int32_t *in_stride,
                                  int32_t *out_stride,
                                  int32_t *input_shape,
                                  int32_t *output_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> dst_shape,
                                  const int32_t ndim) {
    //printf("here2\n");
    int32_t iter = ndim - 1;
    out_stride[iter] = 1;
    in_stride[iter] = 1;
    input_shape[iter] = (int32_t)in_shape[iter];
    output_shape[iter] = (int32_t)dst_shape[iter];
    iter--;
    for (; iter >= 0; --iter) {
      out_stride[iter] = out_stride[iter+1] * dst_shape[iter+1];
      in_stride[iter] = in_stride[iter+1] * in_shape[iter+1];
      input_shape[iter] = (int32_t)in_shape[iter];
      output_shape[iter] = (int32_t)dst_shape[iter];
    }
  }
};

inline void TestShapeCompact(const mxnet::TShape& big, const mxnet::TShape& small,
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

template<typename xpu>
inline void TestComputeImpl(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs,
                                 const mxnet::TShape& small) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  //printf("Started executing broadcast_axis\n");
  mxnet::TShape src_shape, dst_shape;
  TestShapeCompact(outputs[0].shape_, small, &dst_shape, &src_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
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
      if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, OType> out =
          outputs[0].get_with_shape<xpu, 2, OType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, IType> data =
          inputs[0].get_with_shape<xpu, 2, IType>(src_shape.get<2>(), s);
        if(ctx.run_ctx.get_ctx().dev_type == Context::kGPU){
          mshadow::Tensor<xpu, 1, char> workspace =
              ctx.requested.at(0).get_space_typed<xpu, 1, char>
              (mshadow::Shape1(sizeof(int32_t) * dst_shape.ndim() * 4), s);
          char* workspace_curr_ptr = workspace.dptr_;
          int32_t* out_stride = reinterpret_cast<int32_t*>(workspace_curr_ptr);
          int32_t* in_stride =
          reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim());
          int32_t* input_shape =
            reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim() * 2);
          int32_t* output_shape =
            reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim() * 3);
          //printf("here3\n");
          Kernel<comp_offset<1>, xpu>::Launch(s, 1, in_stride, out_stride, input_shape, output_shape, in_shape, out_shape, dst_shape.ndim());
          Kernel<test_kernel_gpu<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, input_shape, output_shape, in_stride, out_stride, req[0], 2);
        } else {
          Kernel<test_kernel<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, in_shape, out_shape, req[0], 2);
        }
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, OType> out =
          outputs[0].get_with_shape<xpu, ndim, OType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, IType> data =
          inputs[0].get_with_shape<xpu, ndim, IType>(src_shape.get<ndim>(), s);
        if(ctx.run_ctx.get_ctx().dev_type == Context::kGPU){
          mshadow::Tensor<xpu, 1, char> workspace =
              ctx.requested.at(0).get_space_typed<xpu, 1, char>
              (mshadow::Shape1(sizeof(int32_t) * dst_shape.ndim() * 4, s);
          char* workspace_curr_ptr = workspace.dptr_;
          int32_t* out_stride = reinterpret_cast<int32_t*>(workspace_curr_ptr);
          int32_t* in_stride =
          reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim());
          int32_t* input_shape =
            reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim() * 2);
          int32_t* output_shape =
            reinterpret_cast<int32_t*>(workspace_curr_ptr + sizeof(int32_t) * dst_shape.ndim() * 3);
          //printf("here4\n");
          Kernel<comp_offset<1>, xpu>::Launch(s, 1, in_stride, out_stride, input_shape, output_shape, in_shape, out_shape, dst_shape.ndim());
          Kernel<test_kernel_gpu<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, input_shape, output_shape, in_stride, out_stride, req[0], ndim);
        } else {
          Kernel<test_kernel<mshadow_op::identity>, xpu>::Launch(
            s, out.shape_.Size(), data.dptr_, out.dptr_, in_shape, out_shape, req[0], ndim);
        }
      }
    });
  });
}

template<typename xpu>
inline void TestCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  TestComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, inputs[0].shape_);
}

}
}
#endif
