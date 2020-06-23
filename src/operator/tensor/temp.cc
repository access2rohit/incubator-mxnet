/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op_value.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */
#include "./temp.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TempAxesParam);

template<typename DType>
void TempAxisKer(DType* src,
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

inline void TempAxisComputeCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const TempAxesParam& param = nnvm::get<TempAxesParam>(attrs.parsed);
  if (param.axis.ndim() == 1 && inputs[0].shape_[param.axis[0]] == 1 && req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      auto dst = outputs[0].dptr<DType>();
      auto src = inputs[0].dptr<DType>();
      index_t outer = inputs[0].shape_.ProdShape(0, param.axis[0]);
      index_t inner = inputs[0].shape_.ProdShape(param.axis[0], inputs[0].shape_.ndim());
      TempAxisKer(src, dst, outer, inner, param.size[0]);
    });
  } else {
    TempComputeImpl<cpu>(attrs, ctx, inputs, req, outputs, inputs[0].shape_);
  }
}

MXNET_OPERATOR_REGISTER_BROADCAST(temp)
.add_alias("temp")
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
.set_attr_parser(ParamParser<TempAxesParam>)
.add_arguments(TempAxesParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", TempAxesShape)
.set_attr<FCompute>("FCompute<cpu>", TempAxisComputeCPU);

}  // namespace op
}  // namespace mxnet
