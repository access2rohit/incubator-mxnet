#include "./test-inl.h"
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(test)
.set_attr<FCompute>("FCompute<gpu>", TestCompute<gpu>);

}  // namespace op
}  // namespace mxnet
