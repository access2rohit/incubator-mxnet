#include "./broadcast_reduce_op.h"
#include "./test-inl.h"

namespace mxnet {
namespace op {

template<typename DType>
void TestAxisKer(DType* src,
                      DType* dst,
                      index_t outer,
                      index_t inner,
                      index_t size) {
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t i = 0; i < outer * size; i++) {
    const index_t m = i / size;
    const index_t n = i % size;
    void* offset = reinterpret_cast<void*>(dst + m * size * inner + n * inner);
    memcpy(offset, reinterpret_cast<void*>(src + m * inner), inner * sizeof (DType));
  }
}

inline void TestComputeCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const BroadcastAxesParam& param = nnvm::get<BroadcastAxesParam>(attrs.parsed);
  if (param.axis.ndim() == 1 && inputs[0].shape_[param.axis[0]] == 1 && req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      auto dst = outputs[0].dptr<DType>();
      auto src = inputs[0].dptr<DType>();
      index_t outer = inputs[0].shape_.ProdShape(0, param.axis[0]);
      index_t inner = inputs[0].shape_.ProdShape(param.axis[0], inputs[0].shape_.ndim());
      TestAxisKer(src, dst, outer, inner, param.size[0]);
    });
  } else {
    TestComputeImpl<cpu>(attrs, ctx, inputs, req, outputs, inputs[0].shape_);
  }
}

MXNET_OPERATOR_REGISTER_BROADCAST(test)
.add_alias("test_axes")
.describe(R"code(Broadcasts the input array over particular axes.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

`broadcast_axes` is an alias to the function `broadcast_axis`.

Example::

   // given x of shape (1,2,1)
   x = [[[ 1.],
         [ 2.]]]

   // broadcast x on on axis 2
   broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                         [ 2.,  2.,  2.]]]
   // broadcast x on on axes 0 and 2
   broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]],
                                                [[ 1.,  1.,  1.],
                                                 [ 2.,  2.,  2.]]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<BroadcastAxesParam>)
.add_arguments(BroadcastAxesParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", BroadcastAxesShape)
.set_attr<FCompute>("FCompute<cpu>", TestComputeCPU)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& n) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});


}
}
