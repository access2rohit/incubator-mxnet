#include "./temp.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(temp)
.set_attr<FCompute>("FCompute<gpu>", TempCompute<gpu>);

}
}
